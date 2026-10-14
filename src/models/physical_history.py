"""
Fiziki Altin Sabah Tahmini: Tarihsel Rejim Saglamligi (2006-2022, KESIF)
=======================================================================
Kullanim (proje kokunden):
  python -m src.data.collect --start 2005-01-01                       # yeni anlik goruntu (2005'ten)
  python -m src.data.build_panel --write --processed-dir data/processed/history
  python -m src.models.physical_history

Soru: Bosluk iliskisi (gece ons/kur hareketinin ertesi gun fiziki kapanisa yansimasi) farkli piyasa
rejimlerinde kararli mi? Bagimsiz sinavdaki (2023-2026) basarisizlik o doneme mi ozgu, yoksa iliski
gecmiste de bozuluyor muydu?

Kurallar:
  - Ana data/processed/market_panel.csv KULLANILMAZ ve DEGISMEZ; genisletilmis panel ayri klasorde.
  - Hedef tarihi >= 2023-01-01 olan hicbir veri okunmaz: fiziki seri ve panel yuklenir yuklenmez kesilir.
  - Modeller, ozellikler ve pencere kurallari physical_eval.py ile AYNI (yeni ayar yok).
  - Tazelik kurali (FINAL_EXAM_PHYS_V1 ile ayni): kur cubugu tarihi == hedef gun ve ons kaynagi >=
    onceki fiziki gun; saglamayan satirlar disarida (eski Yahoo verisindeki bosluklara karsi).
    Bu nedenle 2016-2022 sonuclari ilk gelistirme raporundan biraz farkli olabilir.
  - Rejimler sonuclardan ONCE sabitlendi: 2006-2010, 2011-2015, 2016-2022.
Bu modul KESIF amaclidir; buradaki sonuclar bagimsiz basari olarak sunulamaz.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import physical_eval as pv
from src.models import point_eval as pe
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
START = "2006-01-01"
LOCK = pd.Timestamp(pv.LOCK_START)
ERAS = [("2006-2010", "2006-01-01", "2011-01-01"), ("2011-2015", "2011-01-01", "2016-01-01"),
        ("2016-2022", "2016-01-01", "2023-01-01")]


class HistoryError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "physical_history"


def fresh(df: pd.DataFrame) -> tuple:
    td = pd.to_datetime(df["target_date"])
    stale = df["y"].notna() & ((pd.to_datetime(df["fx_src_next"]) != td) |
                               (pd.to_datetime(df["gold_src_next"]) < pd.to_datetime(df["Date"])))
    return df[~stale].reset_index(drop=True), int(stale.sum())


def era_tables(pred: pd.DataFrame) -> pd.DataFrame:
    td = pd.to_datetime(pred["target_date"])
    rows = []
    for name, a, b in ERAS:
        sub = pred[(td >= a) & (td < b)].reset_index(drop=True)
        if len(sub) < 60:
            rows.append({"era": name, "n": int(len(sub)), "note": "yetersiz gozlem"})
            continue
        comp = pe.comparison_table(sub, pv.REFERENCE)
        for _, r in comp.iterrows():
            rows.append({"era": name, "n": int(len(sub)), "model": r["model"], "improvement_pct": r["improvement_pct"],
                         "ci95_low_TL": r["diff_ci95_low"], "ci95_high_TL": r["diff_ci95_high"],
                         "p_holm": r["p_holm"], "significantly_better": bool(r["significantly_better"])})
    return pd.DataFrame(rows)


def yearly(pred: pd.DataFrame) -> pd.DataFrame:
    year = pd.to_datetime(pred["target_date"]).dt.year
    rows = []
    for y, g in pred.groupby(year):
        p = g["price_t"].to_numpy(float)
        a = np.exp(g["actual_return"].to_numpy(float))
        e_rw = np.abs(p * (a - 1)).mean()
        e_b = np.abs(p * (a - np.exp(g["pred_Bosluk"].to_numpy(float)))).mean()
        e_c = np.abs(p * (a - np.exp(g["pred_Bosluk-Ihtiyatli"].to_numpy(float)))).mean()
        rows.append({"year": int(y), "n": int(len(g)), "bosluk_improvement_pct": 100 * (1 - e_b / e_rw),
                     "ihtiyatli_improvement_pct": 100 * (1 - e_c / e_rw),
                     "beta_gap_median": float(g["beta_gap"].median()),
                     "direction_hit_pct": float(100 * np.mean(np.sign(g["pred_Bosluk"]) == np.sign(g["actual_return"])))})
    return pd.DataFrame(rows)


def run(phys_path: Path = pv.PHYS_PATH, panel_path: Path = HIST_PANEL) -> dict:
    if Path(panel_path).resolve() == Path(pv.PANEL_PATH).resolve():
        raise HistoryError("Ana panel kullanilamaz; genisletilmis paneli data/processed/history altinda olusturun.")
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    phys, panel = phys[phys["Date"] < LOCK], panel[panel["Date"] < LOCK]
    df = pv.build_frame(phys, panel)
    pv.audit_timing(df)
    df, n_stale = fresh(df)
    pred = pv.generate(df, start=START, end=pv.LOCK_START)
    comp = pe.comparison_table(pred, pv.REFERENCE)
    eras, years = era_tables(pred), yearly(pred)
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / "predictions.csv", index=False)
    comp.to_csv(d / "comparison_vs_random_walk.csv", index=False)
    eras.to_csv(d / "eras.csv", index=False)
    years.to_csv(d / "yearly.csv", index=False)
    summary = {"nature": "KESIF - tarihsel saglamlik (hedef < 2023-01-01)",
               "panel_first_date": str(panel["Date"].min().date()),
               "first_target": str(pd.Timestamp(pred["target_date"].min()).date()),
               "last_target": str(pd.Timestamp(pred["target_date"].max()).date()),
               "n_forecasts": int(len(pred)), "stale_rows_excluded": n_stale,
               "years_bosluk_better": int((years["bosluk_improvement_pct"] > 0).sum()), "years": int(len(years)),
               "worst_year": {k: (float(v) if k != "year" else int(v)) for k, v in
                              years.loc[years["bosluk_improvement_pct"].idxmin(), ["year", "bosluk_improvement_pct"]].items()},
               "timing_note": pv.timing_warning(comp)}
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary, "comparison": comp, "eras": eras, "yearly": years}


def print_report(r: dict):
    pd.set_option("display.width", 200)
    s = r["summary"]
    print(f"  {s['nature']}\n  Panel baslangici: {s['panel_first_date']} | hedef {s['first_target']} .. {s['last_target']} | "
          f"{s['n_forecasts']} tahmin | tazelik nedeniyle disarida: {s['stale_rows_excluded']}\n")
    print("  Tum donem, RW'ye gore:")
    print(r["comparison"][["model", "improvement_pct", "diff_ci95_low", "diff_ci95_high", "p_holm", "significantly_better"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Rejimlere gore (RW'ye gore iyilesme %, anlamli mi):")
    e = r["eras"]
    if "model" in e:
        piv = e.dropna(subset=["model"]).assign(cell=lambda x: x["improvement_pct"].map(lambda v: f"{v:+.1f}") +
                                                np.where(x["significantly_better"], "*", " "))
        print(piv.pivot(index="model", columns="era", values="cell").to_string())
        print("    (* = Holm duzeltmeli anlamli iyilesme)")
    print("\n  Yillara gore (Bosluk ve Ihtiyatli iyilesme %, katsayi b, yon isabeti %):")
    print(r["yearly"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n  Bosluk'un RW'yi gectigi yil: {s['years_bosluk_better']}/{s['years']} | en kotu yil: "
          f"{s['worst_year']['year']} (%{s['worst_year']['bosluk_improvement_pct']:+.1f})")
    print(f"  Zamanlama: {s['timing_note']}")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Fiziki sabah tahmini - tarihsel rejim saglamligi (2006-2022).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 78
    print(f"\n{sep}\nFIZIKI ALTIN SABAH TAHMINI - TARIHSEL SAGLAMLIK (KESIF, 2006-2022)\n{sep}")
    try:
        r = run()
    except (FileNotFoundError, HistoryError, pv.PhysicalEvalError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(r)
    print(f"\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
