"""
Geriye Donuk Bagimsiz Sinav: Yalniz Ons Modeli, 2001-2004 (BACK_EXAM_ONS_V1)
============================================================================
Kullanim (proje kokunden):
  python -m src.models.back_exam fetch      # GC=F gunluk 2000-2004 (internet); AYRI klasore yazar
  python -m src.models.back_exam prepare    # BIR KEZ: sinav verisi + on kayit; sonra git commit
  python -m src.models.back_exam run        # TEK SEFER: on kayit commit edildikten sonra

Soru: 2012-2022 gelistirme doneminde (KESIF, gain_decomposition) "yalniz ons" modeli RW'ye gore %5,1 daha az
hata verdi ve bu kazanc Yahoo kur saatinden bagimsizdi. Ayni yontem, HICBIR secimde kullanilmamis
2001-2004 fiziki verisinde de RW'yi geciyor mu?

Model (gain_decomposition ile ayni tanim, kur yok):
  t = fiziki gun, T = sonraki fiziki gun (T - t <= 4 takvim gunu). Hedef y = log P[T] - log P[t].
  g_ham = log(Yahoo GC=F kapanisi, t gunu; IAB/BIST kapanisindan SONRA) - log(IAB USD/ons kapanisi, t).
  g_ons = g_ham - (onceki 20 satirin g_ham medyani; yalnizca gecmis).
  Ons: y_hat = a + b * g_ons. Egilim: y_hat = a (yalniz sabit; TL deger kaybi kaynakli surukleme).
  RW: y_hat = 0. Hata: mutlak TL, |P_t (e^y - e^y_hat)|.
Birincil: genisleyen pencereli walk-forward (63 satirda bir, 1 satir purge, en az 60 egitim satiri), yalnizca
  2000-2004 verisiyle; degerlendirme hedefleri 2001-01-01 .. 2004-12-31.
  Iki karsilastirma: (1) Ons vs RW, (2) Ons vs Egilim. Her biri: DM (Newey-West + HLN, t(n-1), iki yonlu)
  + eslestirilmis blok bootstrap (20, 5000, tohum 20260101); p degerleri Holm (2) ile duzeltilir.
  Hukum (Holm p ile): DAHA_IYI / DAHA_KOTU / PRATIKTE_ESDEGER (+/-%2) / AYIRT_EDILEMEDI.
  Hipotez destegi = IKI karsilastirmada da DAHA_IYI (kazanc yalnizca surukleme degil, gece ons bilgisi).
  Egilim vs RW betimleyici olarak raporlanir.
Ikincil (hukmu degistirmez): gelistirme katsayilari (a, b; hedef 2011-01-01 .. 2022-12-31) ON KAYITTA
  sayi olarak dondurulur ve 2001-2004'e dogrudan uygulanir.
Veri kurallari: Yahoo kapanisi bir onceki Yahoo kapanisina tam esitse bayat sayilir (eksik). Fiyati sifir veya
  negatif satirlar cikarilir. 2005-01-01 ve sonrasi sinav verisine girmez; gelistirme katsayilari icin
  2023-01-01 ve sonrasi HICBIR sekilde okunmaz.
Dosyalar: ham GC=F data/raw/snapshots_backward/ (mevcut arsive dokunulmaz); sinav verisi
  data/processed/back_exam/dataset.csv; sonuc research/back_exam_ons_v1/.
Sinirlar: 2001-2004 farkli bir rejimdir (yuksek enflasyon, 2001 krizi); bu, ayni kosullarda tekrar degil,
  rejimler arasi saglamlik testidir. IAB'nin o donemdeki kapanis saati belgeyle dogrulanmadi.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data import build_dataset as bd
from src.data import collect as co
from src.models import final_exam as fe
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "BACK_EXAM_ONS_V1"
PREREG_PATH = ROOT / "research" / "preregistration" / "back_exam_ons_v1.json"
REGISTRY_PATH = ROOT / "research" / "registry.jsonl"
RESULT_DIR = ROOT / "research" / "back_exam_ons_v1"
DATA_DIR = ROOT / "data" / "processed" / "back_exam"
SNAP_ROOT = ROOT / "data" / "raw" / "snapshots_backward"
PHYS_PATH = ROOT / "data" / "processed" / "physical" / "physical_gold_daily.csv"
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
CODE_FILES = ["src/models/back_exam.py", "src/models/final_exam.py", "src/models/train.py"]

GOLD_RANGE = ("2000-01-01", "2005-01-01")          # bitis HARIC
DATA_END = "2005-01-01"
EVAL_START, EVAL_END = "2001-01-01", "2004-12-31"
DEV_START, LOCK = "2011-01-01", "2023-01-01"
MAX_GAP, BASIS_WIN, MIN_TRAIN = 4, 20, 60
ALPHA, EQUIV_MARGIN_PCT = 0.05, 2.0
BOOT = {"block": 20, "reps": 5000, "seed": 20260101}


class Refused(Exception):
    """Kural geregi reddedildi."""


def norm_md5(data: bytes) -> str:
    return hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest()


def raw_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def latest_snapshot(root: Path = None) -> Path:
    root = Path(root or SNAP_ROOT)
    snaps = sorted(p for p in root.iterdir() if (p / "manifest.json").exists()) if root.exists() else []
    ok = [p for p in snaps if json.loads((p / "manifest.json").read_text(encoding="utf-8"))
          ["series"].get("gold", {}).get("status") == "ok"]
    if not ok:
        raise FileNotFoundError(f"Basarili GC=F goruntusu yok: {root} (once 'fetch').")
    return ok[-1]


def load_gold(snap: Path) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        g = bd.read_yahoo_csv(Path(snap) / "gold.csv", co.SERIES["gold"]["ticker"])
    g = g[["Date", "close"]].dropna(subset=["close"]).copy()
    g["Date"] = pd.to_datetime(g["Date"]).astype("datetime64[ns]")
    return g.sort_values("Date").reset_index(drop=True)


def build_rows(phys: pd.DataFrame, gold: pd.DataFrame, end: str) -> pd.DataFrame:
    """Hedef ve girdi satirlari; tarihi >= end olan hicbir veri kullanilmaz."""
    end = pd.Timestamp(end)
    p = phys[pd.to_datetime(phys["Date"]) < end].dropna(subset=["kapanis_tl_gr", "kapanis_usd_ons"])
    p = p[(p["kapanis_tl_gr"] > 0) & (p["kapanis_usd_ons"] > 0)].sort_values("Date").reset_index(drop=True)
    df = pd.DataFrame({"t": pd.to_datetime(p["Date"]).astype("datetime64[ns]"), "P_t": p["kapanis_tl_gr"].astype(float),
                       "bist_usd_t": p["kapanis_usd_ons"].astype(float)})
    df["T"], df["P_T"] = df["t"].shift(-1), df["P_t"].shift(-1)
    df = df.dropna(subset=["T"])
    df = df[(df["T"] - df["t"]).dt.days <= MAX_GAP]
    g = gold[pd.to_datetime(gold["Date"]) < end].sort_values("Date").copy()
    g["Date"] = pd.to_datetime(g["Date"]).astype("datetime64[ns]")
    g.loc[g["close"].eq(g["close"].shift(1)), "close"] = np.nan          # bayat kapanis = eksik
    df["gold_t"] = df["t"].map(g.set_index("Date")["close"])
    df = df.dropna(subset=["gold_t"]).reset_index(drop=True)
    raw = np.log(df["gold_t"] / df["bist_usd_t"])
    df["g_ons"] = raw - raw.shift(1).rolling(BASIS_WIN, min_periods=BASIS_WIN).median()
    df["y"] = np.log(df["P_T"] / df["P_t"])
    return df.dropna(subset=["g_ons"]).reset_index(drop=True)


def dev_coefficients(phys: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Gelistirme katsayilari: hedef 2011-01-01 .. 2022-12-31. 2023+ satirlar okunmadan atilir."""
    lock = pd.Timestamp(LOCK)
    ph = phys[pd.to_datetime(phys["Date"]) < lock]
    gold = panel.loc[pd.to_datetime(panel["Date"]) < lock, ["Date", "gold_usd_oz"]].rename(columns={"gold_usd_oz": "close"})
    rows = build_rows(ph, gold, LOCK)
    d = rows[rows["T"] >= pd.Timestamp(DEV_START)]
    if d.empty or d["T"].max() >= lock:
        raise Refused("Gelistirme satirlari gecersiz.")
    X = np.column_stack([np.ones(len(d)), d["g_ons"].to_numpy(float)])
    a, b = np.linalg.lstsq(X, d["y"].to_numpy(float), rcond=None)[0]
    return {"a": float(a), "b": float(b), "n": int(len(d)), "first_T": str(d["T"].min().date()),
            "last_T": str(d["T"].max().date())}


# ---------------------------------------------------------------------------
# Walk-forward ve istatistik
# ---------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    cand = df.index[(df["T"] >= pd.Timestamp(EVAL_START)) & (df["T"] <= pd.Timestamp(EVAL_END))].tolist()
    pos = [p for p in cand if p - 1 >= MIN_TRAIN]
    if not pos:
        raise Refused("Degerlendirme satiri yok (egitim satiri yetersiz).")
    out = []
    for w in tr.make_windows(pos, tr.RETRAIN_EVERY):
        ti = np.arange(w[0] - 1)                                       # 1 satir purge
        if df.loc[ti, "T"].max() > df.loc[w[0], "t"]:
            raise Refused("Egitim hedefi degerlendirme gunune tasiyor.")
        X = np.column_stack([np.ones(len(ti)), df.loc[ti, "g_ons"].to_numpy(float)])
        a, b = np.linalg.lstsq(X, df.loc[ti, "y"].to_numpy(float), rcond=None)[0]
        part = df.loc[w, ["t", "T", "P_t", "P_T", "g_ons", "y"]].copy()
        part["pred_Ons"] = a + b * part["g_ons"]
        part["pred_Egilim"] = float(df.loc[ti, "y"].mean())
        part["a"], part["b"], part["n_train"] = a, b, len(ti)
        out.append(part)
    return pd.concat(out).reset_index(drop=True)


def abs_err(pred: pd.DataFrame, yhat) -> np.ndarray:
    p, y = pred["P_t"].to_numpy(float), pred["y"].to_numpy(float)
    return np.abs(p * (np.exp(y) - np.exp(np.asarray(yhat, float))))


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


def compare(e: np.ndarray, e_ref: np.ndarray, p_adj: float = None) -> dict:
    boot = fe.paired_bootstrap(e, e_ref, block=BOOT["block"], reps=BOOT["reps"], seed=BOOT["seed"])
    dm = fe.dm_hln(e, e_ref)
    p = dm["p_value"] if p_adj is None else p_adj
    return {"n": int(len(e)), "MAE_TL": float(e.mean()), "ref_MAE_TL": float(e_ref.mean()),
            "improvement_pct": -boot["pct"], "improvement_ci95": [-boot["pct_ci95"][1], -boot["pct_ci95"][0]],
            "dm_stat_hln": dm["dm_stat_hln"], "p_value": dm["p_value"], "p_used": p,
            "verdict": point_verdict(boot, {"p_value": p})}


def holm(pvals: dict) -> dict:
    out, running = {}, 0.0
    for k, name in enumerate(sorted(pvals, key=pvals.get)):
        running = max(running, min(1.0, pvals[name] * (len(pvals) - k)))
        out[name] = running
    return out


def _pct(e, ref) -> float:
    return float(100 * (1 - e.mean() / ref.mean()))


def evaluate(pred: pd.DataFrame, dev: dict) -> dict:
    e_rw = abs_err(pred, np.zeros(len(pred)))
    e_on = abs_err(pred, pred["pred_Ons"])
    e_eg = abs_err(pred, pred["pred_Egilim"])
    refs = {"Ons_vs_RW": e_rw, "Ons_vs_Egilim": e_eg}
    adj = holm({k: fe.dm_hln(e_on, r)["p_value"] for k, r in refs.items()})
    primary = {k: compare(e_on, r, adj[k]) for k, r in refs.items()}
    hit = float(100 * np.mean(np.sign(pred["pred_Ons"]) == np.sign(pred["y"])))
    frozen = dev["a"] + dev["b"] * pred["g_ons"].to_numpy(float)
    e_fr = abs_err(pred, frozen)
    secondary = {"Donuk_vs_RW": compare(e_fr, e_rw), "Donuk_vs_Egilim": compare(e_fr, e_eg)}
    yrs = []
    for Y, k in pred.groupby(pred["T"].dt.year).groups.items():
        s = pred.loc[k]
        rw, on, eg = abs_err(s, np.zeros(len(s))), abs_err(s, s["pred_Ons"]), abs_err(s, s["pred_Egilim"])
        fr = abs_err(s, dev["a"] + dev["b"] * s["g_ons"])
        yrs.append({"year": int(Y), "n": int(len(s)), "Ons_vs_RW_pct": _pct(on, rw), "Ons_vs_Egilim_pct": _pct(on, eg),
                    "Egilim_vs_RW_pct": _pct(eg, rw), "Donuk_vs_RW_pct": _pct(fr, rw)})
    return {"protocol_id": PROTOCOL_ID, "nature": "BAGIMSIZ (geriye donuk, tek seferlik)",
            "period": [str(pred["T"].min().date()), str(pred["T"].max().date())], "n": int(len(pred)),
            "primary_walk_forward": primary, "direction_hit_pct": hit,
            "descriptive_Egilim_vs_RW": compare(e_eg, e_rw),
            "b_first_last": [float(pred["b"].iat[0]), float(pred["b"].iat[-1])],
            "secondary_frozen_dev": secondary, "yearly": yrs,
            "hypothesis_supported": all(v["verdict"] == "DAHA_IYI" for v in primary.values())}


# ---------------------------------------------------------------------------
# Kayit defteri, on kayit, tek seferlik calistirma
# ---------------------------------------------------------------------------
def _events() -> list:
    path = Path(REGISTRY_PATH)
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("protocol_id") == PROTOCOL_ID]


def _log(event: str, **kw):
    path = Path(REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"protocol_id": PROTOCOL_ID, "event": event, "at_utc": _now(), **kw}) + "\n")


def code_md5() -> dict:
    return {f: norm_md5((Path(ROOT) / f).read_bytes()) for f in CODE_FILES}


def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "gold_range_end_exclusive": list(GOLD_RANGE), "data_end": DATA_END,
            "eval_targets": [EVAL_START, EVAL_END], "max_gap_days": MAX_GAP, "basis_window": BASIS_WIN,
            "min_train": MIN_TRAIN, "retrain_every": tr.RETRAIN_EVERY, "purge": "1 satir",
            "primary": ["Ons vs RW", "Ons vs Egilim (yalniz sabit)"], "loss": "mutlak TL hata",
            "test": "DM Newey-West + HLN, t(n-1), iki yonlu", "multiple": "Holm (2)",
            "supported_if": "iki karsilastirmada da DAHA_IYI (Holm p)",
            "alpha": ALPHA, "equivalence_margin_pct": EQUIV_MARGIN_PCT, "bootstrap": BOOT,
            "secondary": "gelistirme katsayilari donuk (hedef 2011-01-01 .. 2022-12-31)",
            "stale_rule": "Yahoo kapanisi onceki Yahoo kapanisina esitse eksik"}


def prepare(snap: Path = None, phys_path: Path = None, panel_path: Path = None) -> dict:
    prereg_path, data_dir = Path(PREREG_PATH), Path(DATA_DIR)
    if prereg_path.exists():
        raise Refused("On kayit zaten var; sinav verisi ve kurallar degistirilemez.")
    snap = Path(snap or latest_snapshot())
    phys_path, panel_path = Path(phys_path or PHYS_PATH), Path(panel_path or HIST_PANEL)
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    gold = load_gold(snap)
    rows = build_rows(phys, gold, DATA_END)
    dev = dev_coefficients(phys, pd.read_csv(panel_path, parse_dates=["Date"]))
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = data_dir / "dataset.csv"
    rows.to_csv(ds, index=False)
    ev = rows[(rows["T"] >= pd.Timestamp(EVAL_START)) & (rows["T"] <= pd.Timestamp(EVAL_END))]
    counts = {"gold_days": int(len(gold)), "gold_first": str(gold["Date"].min().date()),
              "gold_last": str(gold["Date"].max().date()), "rows": int(len(rows)),
              "eval_candidates": int(len(ev)), "eval_candidates_by_year": {str(k): int(v) for k, v in
                                                                         ev["T"].dt.year.value_counts().sort_index().items()}}
    content = {"created_at_utc": _now(), "parameters": parameters(), "dev_coefficients": dev, "counts": counts,
               "gold_snapshot": snap.name,
               "data_md5": {"dataset.csv": raw_md5(ds), "gold.csv": raw_md5(snap / "gold.csv"),
                            "physical_gold_daily.csv": raw_md5(phys_path), "history_market_panel.csv": raw_md5(panel_path)},
               "code_md5": code_md5()}
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=2)
    prereg_path.write_text(text, encoding="utf-8")
    _log("prepared", prereg_md5=norm_md5(text.encode("utf-8")))
    return content


def git_check(paths: list) -> tuple:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return False, "git deposu yok"
    rel = [str(Path(p).resolve().relative_to(Path(ROOT).resolve())).replace("\\", "/") for p in paths]
    if any(git("ls-files", "--error-unmatch", r).returncode != 0 for r in rel):
        return False, "on kayit veya kod commit edilmemis"
    if git("status", "--porcelain", "--", *rel).stdout.strip():
        return False, "on kayit veya kodda commit edilmemis degisiklik var"
    return True, head.stdout.strip()


def run(git_fn=None) -> dict:
    prereg_path, data_dir, result_dir = Path(PREREG_PATH), Path(DATA_DIR), Path(RESULT_DIR)
    if not prereg_path.exists():
        raise Refused("On kayit yok: once 'prepare' ve commit.")
    if any(e["event"] in ("started", "completed") for e in _events()):
        raise Refused("Sinav zaten baslatildi; tekrar calistirilamaz.")
    pr = json.loads(prereg_path.read_text(encoding="utf-8"))
    if code_md5() != pr["code_md5"]:
        raise Refused("Kod on kayittan farkli.")
    ds = data_dir / "dataset.csv"
    if not ds.exists() or raw_md5(ds) != pr["data_md5"]["dataset.csv"]:
        raise Refused("Sinav verisi on kayittan farkli veya yok.")
    ok, info = (git_fn or git_check)([prereg_path] + [Path(ROOT) / f for f in CODE_FILES])
    if not ok:
        raise Refused(f"Git: {info}")
    _log("started", commit=info)
    df = pd.read_csv(ds, parse_dates=["t", "T"], float_precision="round_trip")
    pred = walk_forward(df)
    res = evaluate(pred, pr["dev_coefficients"])
    result_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(result_dir / "predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(res["yearly"]).to_csv(result_dir / "yearly.csv", index=False, float_format="%.6g")
    text = json.dumps(res, indent=2)
    (result_dir / "result.json").write_text(text, encoding="utf-8")
    _log("completed", result_md5=norm_md5(text.encode("utf-8")), supported=res["hypothesis_supported"])
    return res


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Geriye donuk bagimsiz sinav (BACK_EXAM_ONS_V1).")
    ap.add_argument("command", choices=["fetch", "prepare", "run"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 78
    print(f"\n{sep}\nGERIYE DONUK BAGIMSIZ SINAV - {PROTOCOL_ID} - {args.command.upper()}\n{sep}")
    try:
        if args.command == "fetch":
            m = co.collect(["gold"], *GOLD_RANGE, snapshot_root=Path(SNAP_ROOT))
            g = m["series"]["gold"]
            print(f"  Durum: {g['status']} | goruntu {m['snapshot_dir']}")
            return 0 if g["status"] == "ok" else 1
        if args.command == "prepare":
            c = prepare()
            n, d = c["counts"], c["dev_coefficients"]
            print(f"  GC=F: {n['gold_days']} gun ({n['gold_first']} .. {n['gold_last']}) | sinav satiri {n['rows']}")
            print(f"  Degerlendirme adayi: {n['eval_candidates']} | yillara gore {n['eval_candidates_by_year']}")
            print(f"  Gelistirme katsayilari (donuk, ikincil): a {d['a']:.6f}, b {d['b']:.4f}, n {d['n']} "
                  f"({d['first_T']} .. {d['last_T']})")
            print(f"  On kayit: {Path(PREREG_PATH)}\n  SIMDI: on kaydi, kayit defterini ve kodu commit edin; sonra 'run'.")
            return 0
        r = run()
        print(f"  Donem: {r['period'][0]} .. {r['period'][1]} | n {r['n']} | yon isabeti %{r['direction_hit_pct']:.1f} "
              f"| b {r['b_first_last'][0]:.3f} -> {r['b_first_last'][1]:.3f}")
        rows = [("BIRINCIL", r["primary_walk_forward"]), ("Ikincil", r["secondary_frozen_dev"]),
                ("Betimleyici", {"Egilim_vs_RW": r["descriptive_Egilim_vs_RW"]})]
        for label, block in rows:
            for k, c in block.items():
                print(f"  {label:11s} {k:16s} {c['verdict']:17s} iyilesme %{c['improvement_pct']:+.2f} "
                      f"[{c['improvement_ci95'][0]:+.2f}, {c['improvement_ci95'][1]:+.2f}] | p {c['p_used']:.4f}")
        for y in r["yearly"]:
            print(f"    {y['year']}: n {y['n']} | Ons/RW %{y['Ons_vs_RW_pct']:+.1f} | Ons/Egilim %{y['Ons_vs_Egilim_pct']:+.1f}"
                  f" | Egilim/RW %{y['Egilim_vs_RW_pct']:+.1f} | donuk/RW %{y['Donuk_vs_RW_pct']:+.1f}")
        print(f"  Hipotez destegi: {'EVET' if r['hypothesis_supported'] else 'HAYIR'} | Sonuc: {Path(RESULT_DIR)}")
        return 0
    except Refused as e:
        print(f"  REDDEDILDI: {e}")
        return 3
    except (FileNotFoundError, bd.DataError, KeyError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
