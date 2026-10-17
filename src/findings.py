"""
Arastirma Bulgulari (salt okunur ozet)
=====================================
Kayitli sonuc dosyalarindan (research/exploratory/*/summary.json, research/back_exam_ons_v1/result.json) panoda
gosterilecek ANAHTAR SAYILARI cikarir. Hicbir sey hesaplamaz, tahmin etmez veya yazmaz; metinler panodadir
(src/web/app.js). Dosya yoksa ilgili bulgu "available": False doner.
Kullanim: python -m src.findings   (ozeti yazdirir)
"""

from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "sizinti": "research/exploratory/leakage_demo/summary.json",
    "oynaklik": "research/exploratory/volatility_study/summary.json",
    "bilesen": "research/exploratory/vol_components/summary.json",
    "trend": "research/exploratory/trend_study/summary.json",
    "prim": "research/exploratory/premium_study/summary.json",
    "geriye_donuk": "research/back_exam_ons_v1/result.json",
    "guc": "research/exploratory/power_study/summary.json",
}


def _by(rows: list, model: str) -> dict:
    return next(r for r in rows if r["model"] == model)


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _sizinti(d):
    k1, k3 = d["K1_literatur_rastgele_bolme"], d["K3_dogru_walk_forward"]
    models = [r for r in k3 if r["model"] not in ("Yarin=bugun", "Ortalama")]
    best = max(models, key=lambda r: r["vs_rw_pct"])
    return {"k1_r2_rw": _by(k1, "Yarin=bugun")["r2_level"], "k1_r2_lstm": _by(k1, "LSTM")["r2_level"],
            "k1_lstm_vs_rw_pct": _by(k1, "LSTM")["vs_rw_pct"], "k3_best_model": best["model"],
            "k3_best_vs_rw_pct": best["vs_rw_pct"], "k3_best_dm_p": best["dm_p"],
            "k3_hit_min": min(r["direction_hit_pct"] for r in models), "k3_hit_max": max(r["direction_hit_pct"] for r in models),
            "n": d["K3_n"], "period": f"{d['config']['eval_years'][0]}-{d['config']['eval_years'][-1]}"}


def _oynaklik(d):
    g, s = _by(d["models"], "GARCH"), _by(d["models"], "Sabit")
    years = d["yearly"]
    return {"garch_gain": g["qlike_gain_vs_sabit"], "garch_years_better": sum(y["GARCH_gain"] > 0 for y in years),
            "years": len(years), "garch_christoffersen80_p": g["christoffersen80_p"],
            "sabit_christoffersen80_p": s["christoffersen80_p"], "garch_cov80": g["cov80"], "garch_cov95": g["cov95"],
            "garch_dm_p": g["dm_p_vs_sabit"]}


def _bilesen(d):
    y = {r["year"]: r for r in d["yearly"]}
    b2 = _by(d["vs_garch"], "Birlesik2")
    return {"gvz_2018": y[2018]["GVZ_gain"], "bilesengvz_2018": y[2018]["BilesenGVZ_gain"],
            "gvz_2021": y[2021]["GVZ_gain"], "bilesengvz_2021": y[2021]["BilesenGVZ_gain"],
            "best_vs_garch": max(r["qlike_gain_vs_garch"] for r in d["vs_garch"]),
            "birlesik2_years_better": b2["years_better_than_garch"], "years": len(d["yearly"])}


def _trend(d):
    out = {}
    for h in ("hafta", "ay"):
        r, dr = d[h]["returns"], d[h]["direction"]
        out[h] = {"n": d[h]["n"], "up_rate_pct": d[h]["up_rate_pct"],
                  "ortalama_vs_rw_pct": _by(r, "Ortalama")["vs_rw_pct"], "ortalama_dm_p": _by(r, "Ortalama")["dm_p_vs_rw"],
                  "dogrusal_vs_ortalama_pct": _by(r, "Dogrusal")["vs_ortalama_pct"],
                  "lojistik_brier_skill": _by(dr, "Lojistik")["brier_skill_vs_iklim"],
                  "lojistik_hit_pct": _by(dr, "Lojistik")["hit_pct"], "hep_yukari_hit_pct": _by(dr, "Hep_yukari")["hit_pct"]}
    return out


def _prim(d):
    a, f = d["ar1_weekly"], _by(d["forecast_next_week"], "AR1")
    return {"phi": a["phi"], "phi_ci95": a["phi_ci95"], "half_life_weeks": a["half_life"], "mean_pct": a["mu"],
            "ar1_vs_rw_pct": f["vs_rw_pct"], "ar1_dm_p": f["dm_p_vs_rw"], "weeks": a["n"]}


def _geriye(d):
    p = d["primary_walk_forward"]
    return {"supported": d["hypothesis_supported"], "n": d["n"], "period": d["period"],
            "ons_vs_rw_pct": p["Ons_vs_RW"]["improvement_pct"], "ons_vs_rw_ci": p["Ons_vs_RW"]["improvement_ci95"],
            "ons_vs_rw_p": p["Ons_vs_RW"]["p_used"], "ons_vs_egilim_pct": p["Ons_vs_Egilim"]["improvement_pct"],
            "ons_vs_egilim_p": p["Ons_vs_Egilim"]["p_used"]}


def _guc(d):
    ph, th = d["physical_bosluk_vs_rw"], d["theoretical_mean_vs_rw_proxy"]
    k80 = {round(r["true_coverage_pct"]): r["power"] for r in d["kupiec"]["80"]}
    ch = {r["p_violation_after_violation"]: r["power"] for r in d["christoffersen_80"]}
    return {"n": d["config"]["eval_n"], "phys_mde80": ph["mde80_pct"], "phys_mde80_holm": ph["mde80_holm3_pct"],
            "phys_observed": ph["observed"]["full_period_improvement_pct"],
            "phys_share_sig": ph["observed"]["share_windows_significant"], "phys_size": ph["size_at_zero"],
            "theo_mde80": th["mde80_pct"], "theo_observed": th["observed"]["full_period_improvement_pct"],
            "kupiec80_at70": k80.get(70), "kupiec80_at75": k80.get(75), "christoffersen_at05": ch.get(0.5)}


EXTRACT = {"sizinti": _sizinti, "oynaklik": _oynaklik, "bilesen": _bilesen, "trend": _trend, "prim": _prim,
           "geriye_donuk": _geriye, "guc": _guc}


def collect(root: Path = None) -> list:
    root = Path(root or ROOT)
    items = []
    for key, rel in FILES.items():
        p = root / rel
        if not p.exists():
            items.append({"id": key, "available": False, "source": rel})
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append({"id": key, "available": True, "source": rel, "nature": data.get("nature", ""),
                          "values": _clean(EXTRACT[key](data))})
        except (KeyError, StopIteration, ValueError, TypeError) as e:
            items.append({"id": key, "available": False, "source": rel, "error": f"{type(e).__name__}: {e}"})
    return items


def main(argv=None) -> int:
    for it in collect():
        print(f"[{'VAR' if it['available'] else 'YOK'}] {it['id']}: {json.dumps(it.get('values', it.get('error', '')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
