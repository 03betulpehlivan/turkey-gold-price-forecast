"""
2026 Bagimsiz Son Sinav (FINAL_EXAM_V1)
=======================================
Kullanim (proje kokunden), SIRAYLA:
  1) python -m src.models.final_exam prepare     # veri seti + on kayit (hesaplama YOK)
  2) git add research/preregistration/final_exam_v1.json && git commit -m "FINAL_EXAM_V1 on kayit"
  3) python -m src.models.final_exam run         # TEK SEFERLIK sinav
  4) git add research && git commit -m "FINAL_EXAM_V1 sonuc"

Protokol (onaylandi; kod sabitleri):
  Veri      : anlik goruntu SNAPSHOT; 2015-2025 kismi mevcut model_dataset.csv ile 1e-9
              toleransla ayni olmali. 2026 iceren veri seti AYRI dosyaya yazilir.
  Kume      : HEDEF tarihi 2026 olan tum tahminler (2025'in son gunundeki tahmin dahil).
  Egitim    : 63 gunluk genisleyen pencereler, 1 satir purge (train.py ile ayni).
  Nokta     : Mean-Return (birincil) - Zero-Return (referans). Kayip: mutlak TL hata.
              Sonuc: DAHA_IYI | DAHA_KOTU | PRATIKTE_ESDEGER (+/-%2) | AYIRT_EDILEMEDI
              (HLN duzeltmeli DM, t(n-1), iki yonlu + eslestirilmis blok bootstrap GA).
  Aralik    : aci_gvzlag_g020 (birincil, olcek 2026 basinda 1) - garch (referans).
              A1 Kupiec p > 0.05 (%80 ve %95), A2 Christoffersen p > 0.05 (%80),
              A3 pinball kotu-olmama: bootstrap GA ust ucu < +%2.
              Hepsi gecerse: "hedef olasilikla celisen kanit bulunmadi" (kalibrasyon KANITLANDI denmez).
  Tek sefer : on kayit git'e commit edilmeden calismaz; kod/veri md5'leri on kayitla ayni
              olmali; research/registry.jsonl'de bu protokol icin tamamlanmis sonuc varsa
              (cikti klasorunden bagimsiz) reddedilir; her deneme kayda yazilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.data import build_dataset as bd
from src.data import build_panel as bp
from src.data import collect as co
from src.models import adaptive as ad
from src.models import intervals as iv
from src.models import point_eval as pe
from src.models import train as tr

PROTOCOL_ID = "FINAL_EXAM_V1"
SNAPSHOT = "20261011T174148Z"
EXAM_START = "2026-01-01"
POINT_PRIMARY, POINT_REF = "Mean-Return", "Zero-Return"
INTERVAL_PRIMARY, INTERVAL_REF = "aci_gvzlag_g020", "garch"
EQUIV_MARGIN_PCT = 2.0
NONINF_MARGIN_PCT = 2.0
ALPHA = 0.05
BOOT_BLOCK, BOOT_REPS, BOOT_SEED = 20, 5000, 20260101
TL_THRESHOLDS = (0.40, 1.0, 5.0, 10.0)

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"
PREREG_PATH = RESEARCH_DIR / "preregistration" / "final_exam_v1.json"
REGISTRY_PATH = RESEARCH_DIR / "registry.jsonl"
RESULT_DIR = RESEARCH_DIR / "final_exam_v1"
EXAM_DATA_PATH = ROOT / "data" / "processed" / "final_exam" / "model_dataset_2015_2026.csv"
EXAM_THEO_PATH = ROOT / "data" / "processed" / "final_exam" / "theoretical_2015_2026.csv"
EXISTING_MODEL_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
PANEL_REPORT_PATH = ROOT / "data" / "processed" / "market_panel_report.json"
SNAPSHOT_ROOT = co.DEFAULT_SNAPSHOT_ROOT
CODE_MODULES = [bd, bp, tr, iv, ad, pe]


class ExamError(Exception):
    """Sinav on kosulu saglanmadi; hicbir sonuc uretilmedi."""


class ExamAlreadyRun(ExamError):
    """Bu protokol icin tamamlanmis bir sonuc zaten var."""


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_files() -> dict:
    files = {Path(m.__file__).name: Path(m.__file__) for m in CODE_MODULES}
    files[Path(__file__).name] = Path(__file__)
    return files


# ---------------------------------------------------------------------------
# Kayit defteri ve git
# ---------------------------------------------------------------------------
def registry_entries() -> list:
    if not REGISTRY_PATH.exists():
        return []
    return [json.loads(l) for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def log_event(event: str, **extra):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"protocol_id": PROTOCOL_ID, "event": event, "at_utc": _now(), **extra},
                           ensure_ascii=False) + "\n")


def git_state(paths: list) -> dict:
    """Git durumu: commit, verilen yollarda commit edilmemis degisiklik, on kaydin izlenip izlenmedigi."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"ok": False, "reason": "git deposu bulunamadi"}
    rel = [str(Path(p).resolve().relative_to(ROOT.resolve())) for p in paths]
    dirty = git("status", "--porcelain", "--", *rel).stdout.strip()
    tracked = git("ls-files", "--error-unmatch", str(PREREG_PATH.resolve().relative_to(ROOT.resolve()))).returncode == 0
    return {"ok": True, "commit": head.stdout.strip(), "dirty": dirty, "prereg_tracked": tracked}


GIT = git_state


# ---------------------------------------------------------------------------
# Veri seti (2015-2026, ayri dosya)
# ---------------------------------------------------------------------------
def build_exam_dataset() -> dict:
    snap = SNAPSHOT_ROOT / SNAPSHOT
    try:
        manifest, series = bp.load_snapshot(snap)              # ham dosya md5'leri manifest'le dogrulanir
    except (bp.PanelError, FileNotFoundError) as e:
        raise ExamError(f"Anlik goruntu okunamadi ({snap}): {e}") from e
    theo, model = _quiet(bd.build, series["gold"], series["usdtry"])
    existing = pd.read_csv(EXISTING_MODEL_PATH, parse_dates=["Date"])
    old = model[model["Date"] <= existing["Date"].max()].reset_index(drop=True)
    if list(old["Date"]) != list(existing["Date"]):
        raise ExamError("2015-2025 tarihleri mevcut model_dataset.csv ile ayni degil.")
    for c in existing.columns[1:]:
        if not np.allclose(old[c].to_numpy(float), existing[c].to_numpy(float), rtol=0, atol=1e-9):
            raise ExamError(f"2015-2025 '{c}' sutunu mevcut model_dataset.csv ile ayni degil.")
    out = {}
    for frame, path in ((model, EXAM_DATA_PATH), (theo, EXAM_THEO_PATH)):
        text = frame.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n")
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ExamError(f"{path.name} mevcut ve farkli; sinav verisi degistirilemez.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        out[path.name] = md5(path)
    out["raw"] = {n: e.get("md5") for n, e in manifest["series"].items() if n in ("gold", "usdtry", "gvz")}
    out["rows_2026_target"] = int((theo["Date"] >= pd.Timestamp(EXAM_START)).sum())
    return out


def check_panel() -> dict:
    if not PANEL_REPORT_PATH.exists() or not PANEL_PATH.exists():
        raise ExamError("market_panel.csv veya raporu yok (python -m src.data.build_panel --write).")
    rep = json.loads(PANEL_REPORT_PATH.read_text(encoding="utf-8"))
    if rep.get("snapshot_dir") != SNAPSHOT:
        raise ExamError(f"Panel farkli bir anlik goruntuden kurulmus: {rep.get('snapshot_dir')} != {SNAPSHOT}.")
    if "gvz" not in pd.read_csv(PANEL_PATH, nrows=1).columns:
        raise ExamError("Panelde gvz yok.")
    return {"market_panel.csv": md5(PANEL_PATH)}


def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "snapshot": SNAPSHOT, "exam_start_target_date": EXAM_START,
            "point": {"primary": POINT_PRIMARY, "reference": POINT_REF, "loss": "mutlak TL hata",
                      "equivalence_margin_pct": EQUIV_MARGIN_PCT},
            "interval": {"primary": INTERVAL_PRIMARY, "reference": INTERVAL_REF, "aci_scale_start": 1.0,
                         "noninferiority_margin_pct": NONINF_MARGIN_PCT,
                         "criteria": ["A1 Kupiec p>0.05 (80, 95)", "A2 Christoffersen p>0.05 (80)",
                                      "A3 pinball bootstrap GA ust < +%2"]},
            "alpha": ALPHA, "dm": "Newey-West + HLN duzeltmesi, t(n-1), iki yonlu",
            "bootstrap": {"type": "eslestirilmis hareketli blok", "block": BOOT_BLOCK, "reps": BOOT_REPS,
                          "seed": BOOT_SEED},
            "retrain_every": tr.RETRAIN_EVERY, "purge": "1 satir",
            "excluded": {"Bilesen-Gecikme": "zamanlama teshisi KARARSIZ; secim disi"}}


def prepare() -> dict:
    """Veri setini kurar ve on kaydi yazar. Hesaplama YAPMAZ."""
    data = build_exam_dataset()
    panel = check_panel()
    content = {"parameters": parameters(),
               "code_md5": {n: md5(p) for n, p in code_files().items()},
               "data_md5": {**{k: v for k, v in data.items() if k.endswith(".csv")}, **panel, "raw": data["raw"]},
               "expected_forecasts_approx": data["rows_2026_target"]}
    if PREREG_PATH.exists():
        old = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
        old.pop("created_at_utc", None)
        if old != content:
            raise ExamError("On kayit zaten var ve icerik farkli: protokol degistirilemez (yeni kimlik gerekir).")
        return {"status": "zaten hazir", "path": str(PREREG_PATH), **content}
    PREREG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREREG_PATH.write_text(json.dumps({"created_at_utc": _now(), **content}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    log_event("prepared", prereg_md5=md5(PREREG_PATH))
    return {"status": "olusturuldu", "path": str(PREREG_PATH), **content}


# ---------------------------------------------------------------------------
# Istatistik
# ---------------------------------------------------------------------------
def dm_hln(loss_model, loss_ref) -> dict:
    """Newey-West DM (train.diebold_mariano) + Harvey-Leybourne-Newbold duzeltmesi (h=1), t(n-1)."""
    d = tr.diebold_mariano(np.asarray(loss_model, float), np.asarray(loss_ref, float))
    n = d["n"]
    stat = d["dm_stat"] * math.sqrt((n - 1) / n) if np.isfinite(d["dm_stat"]) else np.nan
    p = float(2 * stats.t.sf(abs(stat), n - 1)) if np.isfinite(stat) else np.nan
    return {"n": n, "mean_loss_diff": d["mean_loss_diff"], "dm_stat_nw": d["dm_stat"], "dm_stat_hln": stat,
            "p_value": p, "hac_lag": d["hac_lag"]}


def paired_bootstrap(loss_model, loss_ref, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED) -> dict:
    """Eslestirilmis hareketli blok bootstrap: ortalama fark ve yuzde degisim icin %95 GA."""
    lm, lr = np.asarray(loss_model, float), np.asarray(loss_ref, float)
    n = len(lm)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    k = int(math.ceil(n / block))
    diff, pct = np.empty(reps), np.empty(reps)
    for i in range(reps):
        idx = (rng.integers(0, n - block + 1, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        a, b = lm[idx].mean(), lr[idx].mean()
        diff[i], pct[i] = a - b, 100 * (a / b - 1)
    return {"diff": float(lm.mean() - lr.mean()), "pct": float(100 * (lm.mean() / lr.mean() - 1)),
            "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "pct_ci95": [float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))]}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def point_verdict(boot: dict, dm: dict) -> str:
    lo, hi = boot["diff_ci95"]
    plo, phi = boot["pct_ci95"]
    if hi < 0 and dm["p_value"] < ALPHA:
        return "DAHA_IYI"
    if lo > 0 and dm["p_value"] < ALPHA:
        return "DAHA_KOTU"
    if -EQUIV_MARGIN_PCT < plo and phi < EQUIV_MARGIN_PCT:
        return "PRATIKTE_ESDEGER"
    return "AYIRT_EDILEMEDI"


def interval_verdict(summary: dict, boot: dict) -> dict:
    a1 = bool(summary["kupiec_80_p"] > ALPHA and summary["kupiec_95_p"] > ALPHA)
    c80 = summary["christoffersen_80_p"]
    a2 = bool(np.isfinite(c80) and c80 > ALPHA)
    a3 = bool(boot["pct_ci95"][1] < NONINF_MARGIN_PCT)
    ok = a1 and a2 and a3
    return {"A1_kupiec": a1, "A2_christoffersen_80": a2, "A3_pinball_noninferior": a3, "all_passed": ok,
            "statement": ("Aralik kapsamlarinin hedef olasiliklarla celistigine dair kanit bulunmadi ve pinball "
                          "kaybi referanstan en fazla %2 kotu (kalibrasyon KANITLANMIS degildir)." if ok else
                          "On kayitli araliga iliskin kosullardan en az biri saglanmadi.")}


# ---------------------------------------------------------------------------
# Tahmin uretimi (sinav kumesi)
# ---------------------------------------------------------------------------
def load_exam_frame(data_path: Path = None, theo_path: Path = None, panel_path: Path = None) -> pd.DataFrame:
    df = _quiet(tr.load_data, data_path or EXAM_DATA_PATH)
    theo = pd.read_csv(theo_path or EXAM_THEO_PATH, parse_dates=["Date"])
    nxt = dict(zip(theo["Date"].iloc[:-1], theo["Date"].iloc[1:]))
    df["target_date"] = df["Date"].map(nxt)
    panel = pd.read_csv(panel_path or PANEL_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    panel["gvz_lag1"] = panel["gvz"].shift(1)
    df["Date"] = df["Date"].astype("datetime64[ns]")
    panel["Date"] = panel["Date"].astype("datetime64[ns]")
    df = df.merge(panel[["Date", "gvz", "gvz_lag1"]], on="Date", how="left")
    if df[["gvz", "gvz_lag1", "target_date"]].isna().any().any():
        raise ExamError("GVZ veya hedef tarihi eksik satir var.")
    return df


def exam_positions(df: pd.DataFrame) -> list:
    pos = df.index[df["target_date"] >= pd.Timestamp(EXAM_START)].tolist()
    if not pos or pos != list(range(pos[0], len(df))):
        raise ExamError("Sinav satirlari verinin sonunda kesintisiz bir blok olusturmuyor.")
    return pos


def generate(df: pd.DataFrame) -> dict:
    pos = exam_positions(df)
    windows = tr.make_windows(pos, tr.RETRAIN_EVERY)
    y = df[tr.TARGET].to_numpy(float)
    mean_pred, audit = [], []
    for w in windows:
        train_df = tr._purge_train(df, w[0])
        if train_df["Date"].iloc[-1] >= df["Date"].iat[w[0]] or \
                df["target_date"].iat[len(train_df) - 1] > df["Date"].iat[w[0]]:
            raise ExamError("Egitim verisi degerlendirme penceresine tasiyor.")
        mean_pred.extend([float(train_df[tr.TARGET].mean())] * len(w))
        audit.append({"train_last_date": str(train_df["Date"].iloc[-1].date()),
                      "train_last_target_date": str(df["target_date"].iat[len(train_df) - 1].date()),
                      "eval_first_date": str(df["Date"].iat[w[0]].date()), "n_eval": len(w)})
    start = str(df["Date"].iat[pos[0]].date())
    ivp = ad.walk_forward(df, start=start, end=None)["predictions"]
    if not np.array_equal(pd.to_datetime(ivp["prediction_date"]).to_numpy(), df["Date"].iloc[pos].to_numpy()):
        raise ExamError("Aralik tahminleri sinav satirlariyla eslesmiyor.")
    pred = pd.DataFrame({"prediction_date": df["Date"].iloc[pos].to_numpy(),
                         "target_date": df["target_date"].iloc[pos].to_numpy(),
                         "price_t": df[tr.PRICE_COL].iloc[pos].to_numpy(), "actual_return": y[pos],
                         f"pred_{POINT_REF}": 0.0, f"pred_{POINT_PRIMARY}": mean_pred})
    for m in (INTERVAL_REF, INTERVAL_PRIMARY):
        for tau in iv.QUANTILES:
            pred[f"{m}_q{tau}"] = ivp[f"{m}_q{tau}"].to_numpy()
    pred["y"] = pred["actual_return"]
    pred["window"] = ivp["window"].to_numpy()
    return {"predictions": pred, "window_audit": audit}


def evaluate(pred: pd.DataFrame) -> dict:
    e_m, e_r = pe.abs_tl_errors(pred, POINT_PRIMARY), pe.abs_tl_errors(pred, POINT_REF)
    pm = {m: pe.point_metrics(pred["actual_return"], pred[f"pred_{m}"], pred["price_t"]) for m in (POINT_REF, POINT_PRIMARY)}
    p_boot, p_dm = paired_bootstrap(e_m, e_r), dm_hln(e_m, e_r)
    l_m, l_r = iv.row_loss(pred, INTERVAL_PRIMARY), iv.row_loss(pred, INTERVAL_REF)
    i_boot, i_dm = paired_bootstrap(l_m, l_r), dm_hln(l_m, l_r)
    isum = {m: iv.summarize(pred, m) for m in (INTERVAL_REF, INTERVAL_PRIMARY)}
    n = len(pred)
    for m, s in isum.items():
        for lvl in iv.LEVELS:
            s[f"coverage_{lvl}_wilson95"] = wilson(int(round(s[f"coverage_{lvl}"] * n)), n)
    monthly = pd.DataFrame([{"month": str(k), "model": m, **{c: v for c, v in pe.point_metrics(
        g["actual_return"], g[f"pred_{m}"], g["price_t"]).items() if c in ("n", "MAE_TL", "MedAE_TL", "MAPE_pct")}}
        for k, g in pred.groupby(pd.to_datetime(pred["target_date"]).dt.to_period("M")) for m in (POINT_REF, POINT_PRIMARY)])
    return {"point": {"metrics": pm, "bootstrap": p_boot, "dm": p_dm, "verdict": point_verdict(p_boot, p_dm)},
            "interval": {"summary": isum, "bootstrap_pinball": i_boot, "dm_pinball": i_dm,
                         "verdict": interval_verdict(isum[INTERVAL_PRIMARY], i_boot)},
            "monthly_point": monthly, "n": n}


# ---------------------------------------------------------------------------
# Tek seferlik calistirma
# ---------------------------------------------------------------------------
def run() -> dict:
    if any(e.get("event") == "completed" for e in registry_entries() if e.get("protocol_id") == PROTOCOL_ID):
        log_event("refused", reason="tamamlanmis sonuc var")
        raise ExamAlreadyRun(f"{PROTOCOL_ID} icin tamamlanmis sonuc zaten var (research/registry.jsonl).")
    if not PREREG_PATH.exists():
        log_event("refused", reason="on kayit yok")
        raise ExamError("On kayit yok. Once: python -m src.models.final_exam prepare")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    g = GIT([PREREG_PATH] + list(code_files().values()))
    if not g.get("ok") or not g.get("prereg_tracked") or g.get("dirty"):
        log_event("refused", reason="git", git=g)
        raise ExamError(f"Git kosulu saglanmadi (on kayit commit edilmeli, kod degisikligi olmamali): {g}")
    now_code = {n: md5(p) for n, p in code_files().items()}
    now_data = {EXAM_DATA_PATH.name: md5(EXAM_DATA_PATH), EXAM_THEO_PATH.name: md5(EXAM_THEO_PATH),
                PANEL_PATH.name: md5(PANEL_PATH)}
    mism = [k for k, v in now_code.items() if prereg["code_md5"].get(k) != v] + \
           [k for k, v in now_data.items() if prereg["data_md5"].get(k) != v]
    if mism or prereg["parameters"] != parameters():
        log_event("refused", reason="md5/parametre uyusmazligi", files=mism)
        raise ExamError(f"Kod/veri/parametreler on kayitla ayni degil: {mism or 'parametreler'}")
    log_event("started", commit=g["commit"], prereg_md5=md5(PREREG_PATH))

    df = load_exam_frame()
    alignment = pe.verify_price_alignment(df)
    gen = generate(df)
    pred = gen["predictions"]
    if pd.to_datetime(pred["target_date"]).min() < pd.Timestamp(EXAM_START):
        raise ExamError("Sinav kumesine 2026 oncesi hedef girdi.")
    res = evaluate(pred)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(RESULT_DIR / "predictions.csv", index=False)
    res["monthly_point"].to_csv(RESULT_DIR / "monthly_point.csv", index=False)
    result = {"protocol_id": PROTOCOL_ID, "created_at_utc": _now(), "git_commit": g["commit"],
              "prereg_md5": md5(PREREG_PATH), "n_forecasts": res["n"],
              "first_target": str(pd.Timestamp(pred["target_date"].min()).date()),
              "last_target": str(pd.Timestamp(pred["target_date"].max()).date()),
              "alignment_check": alignment, "window_audit": gen["window_audit"],
              "point": res["point"], "interval": res["interval"],
              "environment": {"python": platform.python_version(),
                              **{p: _version(p) for p in ("numpy", "pandas", "scipy", "arch", "scikit-learn")}}}
    (RESULT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float),
                                            encoding="utf-8")
    log_event("completed", result_md5=md5(RESULT_DIR / "result.json"), commit=g["commit"],
              point_verdict=res["point"]["verdict"], interval_all_passed=res["interval"]["verdict"]["all_passed"])
    return result


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def print_result(r: dict):
    p, i = r["point"], r["interval"]
    print(f"  Sinav kumesi: hedef {r['first_target']} .. {r['last_target']} | {r['n_forecasts']} tahmin | commit {r['git_commit'][:10]}")
    print("\n  KARNE 1 - NOKTA TAHMINI (TL)")
    for m in (POINT_REF, POINT_PRIMARY):
        x = p["metrics"][m]
        shares = " ".join(f"<={t:g}TL %{x[f'share_err_le_{t:g}TL_pct']:.1f}" for t in TL_THRESHOLDS)
        print(f"    {m:12s} MAE {x['MAE_TL']:.2f} | MedAE {x['MedAE_TL']:.2f} | RMSE {x['RMSE_TL']:.2f} | "
              f"MAPE %{x['MAPE_pct']:.3f} | bias {x['bias_TL']:+.2f} | yon %{x['direction_accuracy_pct'] or float('nan'):.1f} | {shares}")
    b, d = p["bootstrap"], p["dm"]
    print(f"    Fark (Mean - Zero): {b['diff']:+.3f} TL [{b['diff_ci95'][0]:+.3f}, {b['diff_ci95'][1]:+.3f}] | "
          f"%{b['pct']:+.2f} [{b['pct_ci95'][0]:+.2f}, {b['pct_ci95'][1]:+.2f}] | DM-HLN {d['dm_stat_hln']:+.2f}, p={d['p_value']:.3f}")
    print(f"    SONUC: {p['verdict']}")
    print("\n  KARNE 2 - ARALIK TAHMINI")
    for m in (INTERVAL_REF, INTERVAL_PRIMARY):
        s = i["summary"][m]
        print(f"    {m:16s} %80 kapsam {s['coverage_80']:.3f} {[round(v, 3) for v in s['coverage_80_wilson95']]} | "
              f"%95 kapsam {s['coverage_95']:.3f} {[round(v, 3) for v in s['coverage_95_wilson95']]} | "
              f"genislik80 {s['width_80_TL']:.1f} TL, genislik95 {s['width_95_TL']:.1f} TL | "
              f"Kupiec {s['kupiec_80_p']:.2f}/{s['kupiec_95_p']:.2f} | Chr80 {s['christoffersen_80_p']:.2f}")
    ib = i["bootstrap_pinball"]
    print(f"    Pinball degisimi: %{ib['pct']:+.2f} [{ib['pct_ci95'][0]:+.2f}, {ib['pct_ci95'][1]:+.2f}]")
    v = i["verdict"]
    print(f"    A1 {v['A1_kupiec']} | A2 {v['A2_christoffersen_80']} | A3 {v['A3_pinball_noninferior']}")
    print(f"    SONUC: {v['statement']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2026 bagimsiz son sinav (FINAL_EXAM_V1).")
    ap.add_argument("command", choices=["prepare", "run"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\n2026 BAGIMSIZ SON SINAV - {PROTOCOL_ID} - {args.command.upper()}\n{sep}")
    try:
        if args.command == "prepare":
            r = prepare()
            print(f"  On kayit: {r['status']} -> {r['path']}")
            print(f"  Beklenen tahmin sayisi (yaklasik): {r['expected_forecasts_approx']}")
            print("  SIMDI: git add research/preregistration/final_exam_v1.json && git commit -m \"FINAL_EXAM_V1 on kayit\"")
            return 0
        r = run()
        print_result(r)
        print(f"\n  Ciktilar: {RESULT_DIR}\n  SIMDI: git add research && git commit -m \"FINAL_EXAM_V1 sonuc\"")
        return 0
    except ExamAlreadyRun as e:
        print(f"  REDDEDILDI: {e}")
        return 2
    except ExamError as e:
        print(f"  HATA: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
