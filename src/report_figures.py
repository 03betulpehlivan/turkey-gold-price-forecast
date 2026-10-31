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


def fig_yarisma(plt, out):
    d = _load("research/tool/yarisma_1-3-5_chronos.json")
    P = 6560.0
    h = ["1", "3", "5"]
    sabit = [d["ufuklar"][x]["final"]["Sabit aralik"]["genislik80"] / 100 * P for x in h]
    biz = [d["ufuklar"][x]["final"][d["ufuklar"][x]["kazanan"]]["genislik80"] / 100 * P for x in h]
    t80 = [d["ufuklar"][x]["final"][d["ufuklar"][x]["kazanan"]]["tutma80"] for x in h]      # cubuklar %80 aralik: yazi da %80
    s80 = [d["ufuklar"][x]["final"]["Sabit aralik"]["tutma80"] for x in h]
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = range(len(h))
    ax.bar([i - 0.2 for i in xs], sabit, 0.38, color=C["acik"], label="Basit y\u00f6ntem (sabit aral\u0131k)")
    ax.bar([i + 0.2 for i in xs], biz, 0.38, color=C["iyi"], label="Bu sistem")
    for i in xs:
        ax.text(i - 0.2, sabit[i] + 8, f"{sabit[i]:.0f} TL", ha="center", fontsize=9)
        ax.text(i + 0.2, biz[i] + 8, f"{biz[i]:.0f} TL", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + 0.2, biz[i] / 2, f"%{t80[i]:.1f}\ntuttu".replace(".", ","), ha="center", va="center",
                fontsize=9, color="white")
        ax.text(i - 0.2, sabit[i] / 2, f"%{s80[i]:.1f}\ntuttu".replace(".", ","), ha="center", va="center",
                fontsize=9, color=C["notr"])
    ax.set_xticks(list(xs), ["1 g\u00fcn sonra", "3 g\u00fcn sonra", "5 g\u00fcn sonra"])
    ax.set_ylabel("%80 aral\u0131\u011f\u0131n ortalama geni\u015fli\u011fi (TL)")
    ax.set_title("S\u0131nav d\u00f6nemi 2023\u20132026: ayn\u0131 \u201c%80\u201d iddias\u0131yla daha dar aral\u0131k")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, -0.16, "Gram fiyat\u0131 6.560 TL al\u0131narak hesapland\u0131. \u00c7ubuk i\u00e7indeki say\u0131: %80 aral\u0131\u011f\u0131n ger\u00e7ekte tuttu\u011fu g\u00fcnler.",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["notr"])
    fig.tight_layout()
    fig.savefig(out / "yarisma.png")
    plt.close(fig)


def fig_kriz(plt, out):
    d = _load("research/tool/yarisma_1-3-5_chronos.json")
    u = d["ufuklar"]["1"]
    biz = {r["yil"]: r for r in u["yillar"][u["kazanan"]]}
    sab = {r["yil"]: r for r in u["yillar"]["Sabit aralik"]}
    yil = sorted(biz)
    kriz = {2018: "TL krizi", 2020: "pandemi", 2021: "kur krizi", 2023: "prim\nk\u0131r\u0131lmas\u0131"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for ax in (a1, a2):
        for y in kriz:
            ax.axvspan(y - 0.5, y + 0.5, color="#f1e3c8", zorder=0)
    a1.plot(yil, [sab[y]["genislik80"] for y in yil], "o-", color=C["notr"], label="Basit y\u00f6ntem (sabit aral\u0131k)")
    a1.plot(yil, [biz[y]["genislik80"] for y in yil], "o-", color=C["iyi"], lw=2.4, label="Bu sistem")
    a1.set_ylabel("%80 aral\u0131\u011f\u0131n\ngeni\u015fli\u011fi (%)")
    a1.set_title("Ertesi g\u00fcn tahmini: aral\u0131k krizde geni\u015fliyor, kriz ge\u00e7ince daral\u0131yor")
    a1.legend(frameon=False, loc="upper left")
    for y, ad in kriz.items():
        a1.text(y, a1.get_ylim()[1] * 0.97, ad, ha="center", va="top", fontsize=8, color=C["notr"])
    a2.axhline(80, color="black", lw=1, ls="--")
    a2.plot(yil, [sab[y]["tutma80"] for y in yil], "o-", color=C["notr"])
    a2.plot(yil, [biz[y]["tutma80"] for y in yil], "o-", color=C["iyi"], lw=2.4)
    a2.set_ylabel("%80 aral\u0131k\ntuttu (%)")
    a2.set_ylim(65, 100)
    a2.text(2011.6, 80.8, "hedef %80", fontsize=8)
    a2.axvline(2022.5, color=C["notr"], lw=1)
    a2.text(2022.6, 97, "s\u0131nav d\u00f6nemi \u2192", fontsize=8, color=C["notr"])
    a2.text(2022.4, 97, "\u2190 haz\u0131rl\u0131k d\u00f6nemi", fontsize=8, color=C["notr"], ha="right")
    a2.set_xticks(yil)
    a2.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "kriz.png")
    plt.close(fig)


FIGURES = {"yarisma": fig_yarisma, "kriz": fig_kriz, "sinavlar": fig_exams, "sizinti": fig_leakage, "oynaklik": fig_volatility, "prim": fig_premium, "guc": fig_power}


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
