"""
Rapor Grafikleri
================
Kullanim (proje kokunden):  python -m src.report_figures
Cikti: docs/figures/*.png  (danisman raporu ve README bu dosyalari gosterir)

Grafikler YALNIZCA depodaki kayitli sonuc dosyalarindan (research/...) cizilir; veri indirmez, model
calistirmaz. Bu yuzden her grafik, sonuc dosyalariyla birlikte yeniden uretilebilir.
Gerekli paket: matplotlib (requirements.txt).
"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
C = {"iyi": "#2f6f5e", "kotu": "#9b3b2e", "notr": "#5b6770", "vurgu": "#8a6d22", "acik": "#c9cfc8"}


def _load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})
    return plt


def fig_exams(plt, out):
    v1 = _load("research/final_exam_v1/result.json")["point"]["bootstrap"]
    ph = _load("research/final_exam_phys_v1/result.json")["primary"]["bootstrap"]
    be = _load("research/back_exam_ons_v1/result.json")["primary_walk_forward"]["Ons_vs_RW"]
    rows = [("Teorik fiyat, 2026\n(FINAL_EXAM_V1, 194 g\u00fcn)", -v1["pct"], [-v1["pct_ci95"][1], -v1["pct_ci95"][0]], "Pratikte e\u015fde\u011fer"),
            ("Fiziki fiyat, 2023\u20132026\n(FINAL_EXAM_PHYS_V1, 874 g\u00fcn)", -ph["pct"], [-ph["pct_ci95"][1], -ph["pct_ci95"][0]], "Daha k\u00f6t\u00fc"),
            ("Gece ons bilgisi, 2001\u20132004\n(BACK_EXAM_ONS_V1, 497 g\u00fcn)", be["improvement_pct"], be["improvement_ci95"], "Ay\u0131rt edilemedi")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(-2, 2, color=C["acik"], alpha=0.5, lw=0)
    ax.axvline(0, color=C["notr"], lw=1)
    for i, (lab, m, ci, hukum) in enumerate(rows):
        col = C["kotu"] if ci[1] < 0 else (C["iyi"] if ci[0] > 0 else C["notr"])
        ax.plot(ci, [i, i], color=col, lw=3)
        ax.plot(m, i, "o", color=col, ms=9)
        ax.text(ci[1] + 0.6, i, f"{m:+.1f}%  [{ci[0]:+.1f}, {ci[1]:+.1f}]  {hukum}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-24, 24)
    ax.set_xlabel("\u201cYar\u0131n = bug\u00fcn\u201de g\u00f6re iyile\u015fme (%), %95 g\u00fcven aral\u0131\u011f\u0131   (gri bant: \u00b1%2 pratik e\u015fde\u011ferlik)")
    ax.set_title("Ba\u011f\u0131ms\u0131z s\u0131navlar: \u00f6nceden kay\u0131tl\u0131, tek seferlik")
    fig.tight_layout()
    fig.savefig(out / "sinavlar.png")
    plt.close(fig)


def fig_leakage(plt, out):
    d = _load("research/exploratory/leakage_demo/summary.json")
    k1 = {r["model"]: r for r in d["K1_literatur_rastgele_bolme"]}
    k3 = {r["model"]: r for r in d["K3_dogru_walk_forward"]}
    models = ["Yarin=bugun", "Dogrusal", "XGBoost", "LSTM"]
    names = ["Yar\u0131n = bug\u00fcn", "Do\u011frusal", "XGBoost", "LSTM"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(names, [k1[m]["r2_level"] for m in models], color=[C["notr"]] + [C["vurgu"]] * 3)
    a1.set_ylim(0.99, 1.0)
    a1.set_title("Literat\u00fcr kurulumu: fiyat seviyesinde R\u00b2")
    for i, m in enumerate(models):
        a1.text(i, k1[m]["r2_level"] + 0.0002, f"{k1[m]['r2_level']:.4f}", ha="center", fontsize=9)
    vals = [k1[m]["vs_rw_pct"] for m in models[1:]] + [k3[m]["vs_rw_pct"] for m in models[1:]]
    x = range(len(vals))
    a2.bar(x, vals, color=[C["kotu"] if v < 0 else C["iyi"] for v in vals])
    a2.axhline(0, color=C["notr"], lw=1)
    a2.set_xticks(list(x), [f"{n}\n(literat\u00fcr)" for n in names[1:]] + [f"{n}\n(do\u011fru)" for n in names[1:]], fontsize=8)
    a2.set_ylabel("\u201cYar\u0131n = bug\u00fcn\u201de g\u00f6re (%)")
    a2.set_title("Ayn\u0131 modeller, ger\u00e7ek kazan\u00e7")
    fig.suptitle("Y\u00fcksek R\u00b2 yan\u0131lt\u0131c\u0131: hi\u00e7bir model \u201cyar\u0131n = bug\u00fcn\u201d\u00fc anlaml\u0131 bi\u00e7imde ge\u00e7miyor (fiziki alt\u0131n, 2006\u20132022)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sizinti.png")
    plt.close(fig)


def fig_volatility(plt, out):
    d = _load("research/exploratory/vol_components/summary.json")
    yrs = [y["year"] for y in d["yearly"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.27
    for k, (key, lab, col) in enumerate((("GARCH_gain", "GARCH", C["iyi"]), ("GVZ_gain", "GVZ (yaln\u0131z alt\u0131n)", C["kotu"]),
                                         ("BilesenGVZ_gain", "GVZ + kur bile\u015feni", C["vurgu"]))):
        ax.bar([y + (k - 1) * w for y in yrs], [r[key] for r in d["yearly"]], w, label=lab, color=col)
    ax.axhline(0, color=C["notr"], lw=1)
    ax.set_xticks(yrs)
    ax.set_ylabel("QLIKE kazanc\u0131 (sabit oynakl\u0131\u011fa g\u00f6re)")
    ax.set_title("Oynakl\u0131k tahmin edilebiliyor (GARCH)\nGVZ'nin 2018 ve 2021 TL krizlerindeki ba\u015far\u0131s\u0131zl\u0131\u011f\u0131 kur bile\u015feniyle d\u00fczeliyor")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "oynaklik.png")
    plt.close(fig)


def fig_premium(plt, out):
    d = _load("research/exploratory/premium_study/summary.json")
    y = [r["year"] for r in d["yearly"]]
    med = [r["median"] for r in d["yearly"]]
    iqr = [r["iqr"] for r in d["yearly"]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.errorbar(y, med, yerr=[i / 2 for i in iqr], fmt="o", color=C["vurgu"], ecolor=C["acik"], elinewidth=6, capsize=0)
    ax.axhline(0, color=C["notr"], lw=1)
    a = d["ar1_weekly"]
    ax.set_title(f"Fiziki prim 2006\u20132022: s\u0131f\u0131r civar\u0131nda, yar\u0131 \u00f6m\u00fcr {str(round(a['half_life'], 1)).replace('.', ',')} hafta (2023+ kal\u0131c\u0131 prim emsalsiz)")
    ax.set_ylabel("Prim (%), y\u0131ll\u0131k medyan \u00b1 \u00c7AA/2")
    ax.set_xticks(y)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "prim.png")
    plt.close(fig)


def fig_power(plt, out):
    d = _load("research/exploratory/power_study/summary.json")
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, lab, col in (("physical_bosluk_vs_rw", "Fiziki: Bo\u015fluk vs yar\u0131n = bug\u00fcn", C["vurgu"]),
                          ("theoretical_mean_vs_rw_proxy", "Teorik (vekil): Ortalama getiri vs yar\u0131n = bug\u00fcn", C["iyi"])):
        cur = d[key]["curve"]
        ax.plot([c["effect_pct"] for c in cur], [c["power"] for c in cur], "o-", color=col, label=lab)
    ph = d["physical_bosluk_vs_rw"]["observed"]["full_period_improvement_pct"]
    ax.axhline(0.8, color=C["notr"], ls="--", lw=1)
    ax.axvline(ph, color=C["vurgu"], ls=":", lw=1.5)
    ax.text(ph + 0.5, 0.1, f"geli\u015ftirme d\u00f6neminde\ng\u00f6zlenen etki %{ph:.0f}", color=C["vurgu"], fontsize=9)
    ax.text(41, 0.81, "g\u00fc\u00e7 0,80", fontsize=9, color=C["notr"], ha="right", va="bottom")
    ax.set_xlabel("Ger\u00e7ek iyile\u015fme (%)")
    ax.set_ylabel("120 g\u00fcnde fark edilme olas\u0131l\u0131\u011f\u0131")
    ax.set_title("G\u00fc\u00e7 analizi: 120 tahmin, fiziki tarafta \u2265%20 iyile\u015fmeyi g\u00fcvenle g\u00f6r\u00fcr")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "guc.png")
    plt.close(fig)


FIGURES = {"sinavlar": fig_exams, "sizinti": fig_leakage, "oynaklik": fig_volatility, "prim": fig_premium, "guc": fig_power}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rapor grafikleri (kayitli sonuclardan).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    for name, fn in FIGURES.items():
        fn(plt, out)
        print(f"  yazildi: {out / (name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
