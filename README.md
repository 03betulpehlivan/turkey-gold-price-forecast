# Turkey Gold Price Forecast

Machine learning-based forecasting and analysis of **physical** and **theoretical** gram gold prices in Turkey, built on ounce gold, USD/TRY, market indicators and official TCMB EVDS data, with a strong emphasis on **leakage-free, pre-registered and independently tested evaluation**.

> **Status:** Work in progress (graduation project, Selçuk University, Computer Engineering).
> Research, data pipeline and live forecasting are running; the database, REST API and web interface are planned.
> 📄 **Detailed interim report (Turkish):** [docs/danisman_raporu.md](docs/danisman_raporu.md)

> **Türkçe özet:** Proje, Türkiye'deki fiziki gram altın fiyatı (Borsa İstanbul, TCMB EVDS, 1995–2026) ile ons × USD/TRY'den hesaplanan teorik fiyat için kısa vadeli tahmin ve tahmin aralıkları üretir. Bütün sonuçlar kronolojik, sızıntısız testlerle ve önceden kayıtlı, tek seferlik bağımsız sınavlarla değerlendirilir. Temel bulgular: (1) Günlük fiyat seviyesini "yarın = bugün" yönteminden anlamlı biçimde daha iyi tahmin etmek iki bağımsız sınavda da başarılamadı. Teorik fiyatta model bu yöntemle pratikte eşdeğer çıktı; fiziki fiyatta 2023–2026 sınavında daha kötü sonuç verdi. (2) Borsa İstanbul kapandıktan sonraki gece ons hareketi, ertesi günün fiziki fiyatına yansıyor. 2012–2022 geliştirme döneminde bu bilgi, veri zamanlamasından bağımsız olarak %5 daha az hata sağladı; kur bilgisiyle birlikte kazanç %22. Ancak daha önce hiç kullanılmamış 2001–2004 verisindeki geriye dönük bağımsız sınavda yalnız ons kazancı doğrulanamadı (+%0,5, anlamsız). (3) Yaygın veri kaynağında ciddi zamanlama sorunları tespit edildi. (4) Sistem 15 Ekim 2026'dan itibaren canlı tahminleri değiştirilemez biçimde kaydediyor.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Live System](#live-system)
- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Literature Review](#literature-review)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

Gram gold is one of the most widely held savings instruments in Turkey. Its price is driven mainly by the global ounce gold price and the USD/TRY exchange rate, but the domestic physical market can trade at a premium or discount. This project keeps two price series **separate**:

| | Theoretical gram price | Physical gram price |
|---|---|---|
| Definition | ounce (USD) × USD/TRY / 31.1035 | Daily close of the Borsa Istanbul precious metals market (TL/kg / 1000) |
| Source | Yahoo Finance | TCMB EVDS (Istanbul Gold Exchange archive 1995–2018, Borsa Istanbul 2018–) |
| Coverage | 2005 – today | 1995-07 – today (7,039 trading days) |

The physical price is a **wholesale** market price, not a jeweller's buy/sell price.

## Key Findings

Every result is labelled as **exploratory** (development data; model selection happens here), **independent** (pre-registered, one-shot, unseen data) or **live** (prospective record).

### 1. Theoretical price, next-day point forecasts

| Evaluation | Type | Result |
|---|---|---|
| 2018-01 – 2023-10, 6 models and 8 factor groups | Exploratory | No model significantly better than "tomorrow = today" (RW); RW MAPE 1.07% |
| **2026, 194 forecasts (`FINAL_EXAM_V1`)** | **Independent** | Mean-Return MAPE 1.345% vs. RW 1.347%: **practically equivalent** (difference −0.21% [−1.34%, +1.17%]) |

### 2. Theoretical price, prediction intervals (GARCH, GVZ, adaptive conformal calibration)

| Evaluation | Type | Result |
|---|---|---|
| 2023-10 – 2025-12 confirmation | **Independent** | 80% / 95% coverage 0.788 / 0.940; **hypothesis supported** |
| 2026 | **Independent** | Coverage 0.794 / 0.938, but 80% violations clustered (Christoffersen p ≈ 0.05); **not supported** |

### 3. Physical price, morning nowcast of the day's close

| Evaluation | Type | Result |
|---|---|---|
| 2016–2022, 1,741 forecasts | Exploratory | "Gap" model 22.8% lower error than RW; direction hit rate 71% |
| **2023-01 – 2026-10, 874 forecasts (`FINAL_EXAM_PHYS_V1`)** | **Independent** | Gap model **9.1% worse than RW** [−20.8%, −0.3%], p = 0.03: **not supported**. Direction hit rate 65%. Main explanation: a domestic premium regime in 2023–2024 (physical gold traded 1–5% above the international price) |
| 2012–2022, source of the gain | Exploratory | Overnight **gold** move alone: **+5.1%** [+3.0%, +7.6%], 11 of 11 years, **independent of FX timestamp assumptions**. With overnight FX: +21.7% [+17.9%, +25.2%] |
| **2001-04 – 2004-12, 497 forecasts (`BACK_EXAM_ONS_V1`)** | **Independent** (backward, never-used data) | Overnight-gold model vs RW **+0.5%** [−3.8%, +5.1%], Holm p = 0.81; vs drift-only model +3.0% [−0.6%, +7.3%], p = 0.29: **not supported**. Year-to-year results inconsistent. The sparse 2001–2004 exchange archive (~170 days/year) limited power |

Volatility-scaled intervals with adaptive calibration (2012–2022, exploratory) reached 80% / 95% coverage of 0.800 / 0.949 and removed most violation clustering. Their Winkler score was 27% better than static RW intervals.

### 4. Data findings

- **Timestamp mismatch in Yahoo daily data:** the gold close corresponds to ~18:00 UTC, the USD/TRY close to ~02:00–05:00 UTC. The usual "theoretical gram price" multiplies prices taken 13–16 hours apart. A synchronized hourly pipeline was built.
- **Look-ahead in 2005–2010 FX data:** tested against the FX rate implied by Borsa Istanbul's own simultaneous TL and USD closes. Yahoo's daily USD/TRY bar belonged to a time **after** the Borsa Istanbul close in 2005–2010 (lead correlation 0.25 vs. band 0.06), so those years were excluded from training.
- **Futures roll artifacts:** the continuous gold futures series (GC=F) rolls contracts on different days in hourly and daily data, creating artificial jumps of ~1.5–1.9%.
- **EVDS 1,000-row limit:** responses are silently truncated; downloads are split automatically.

## Live System

| Protocol | What it records | Status |
|---|---|---|
| `LIVE_PHYS_V1` | Physical close forecast each morning: RW, Gap, Gap-Rolling, Premium-Anchor | First forecast 2026-10-15 |
| `LIVE_PHYS_INT_V1` | 80% / 95% intervals around each model; calibrated by replaying the live code with **real hourly FX bars** | First intervals 2026-10-15 |
| `LIVE_V1` | Evening forecast of the synchronized theoretical price | First forecast 2026-10-15; evaluation rule pre-registered before the first outcome |

- **Pre-registration:** rules, models and code hashes were committed before the first forecast.
- **Forecast window:** forecasts are accepted only within a fixed time window on the target day; stale or missing data is refused with a reason.
- **Integrity:** records are append-only and integrity-checked per row, scored automatically when the realized price is published, and committed to git.
- **Automation:** Windows Task Scheduler on weekdays at 12:30 (retry 13:30) and 22:30. Verified with a manually triggered run and the first unattended evening run on 2026-10-15. Each run also copies the untracked data to a backup folder.
- **Evaluation:** a one-shot, pre-registered evaluation after 120 scored forecasts (expected around April 2027).

## Methodology

- **Chronological walk-forward evaluation:** expanding window, retraining every 63 trading days, 1-row purge; no random splits.
- **Leakage controls:** explicit timestamps for every input, truncation-invariance tests, and invariance tests that corrupt all information unknown at forecast time.
- **Pre-registration:** hypotheses, decision rules, code and data hashes are written before each confirmatory run. Confirmatory runs execute once; every attempt is logged in `research/registry.jsonl`.
- **Statistics:**
  - Point forecasts: Diebold-Mariano test (HLN small-sample correction), Holm correction for multiple comparisons, paired moving-block bootstrap confidence intervals.
  - Intervals: Kupiec and Christoffersen tests, Winkler score.
- **Software quality:** 477 unit tests on synthetic data, all passing; GitHub Actions runs them on every code change. Mutation checks verify that the tests detect deliberate errors. Tests do not prove market performance.

## Tech Stack

| Area | Technologies |
|------|--------------|
| Language | Python 3.11 |
| Data | pandas, NumPy, yfinance, TCMB EVDS API (requests) |
| Modelling | scikit-learn, XGBoost, arch (GARCH), SciPy |
| Automation | PowerShell, Windows Task Scheduler, git |
| Testing | unittest |
| Planned | Microsoft SQL Server, REST API (FastAPI), web interface, LSTM benchmark |

## Project Structure

```
turkey-gold-price-forecast/
├── data/                  # raw snapshots and processed data (not tracked by git)
├── docs/                  # interim report, data platform and status notes
├── research/              # pre-registrations, registry, independent test results, live logs
├── scripts/
│   ├── sabah_fiziki.ps1       # morning physical forecast + intervals
│   ├── gunluk_tahmin.ps1      # evening theoretical forecast
│   ├── zamanli_calistir.ps1   # scheduled-run wrapper with logging
│   └── gorev_kur.ps1          # registers the Windows scheduled tasks
├── src/
│   ├── data/              # Yahoo and EVDS collection, validation, panels, physical and synchronized series
│   ├── analysis/          # diagnostics: timing, error analysis, gain decomposition
│   ├── models/            # point and interval models, independent exams, backtests
│   └── live/              # live forecasting, intervals and scoring
├── tests/                 # unit tests (synthetic data)
├── Literatur_Taramasi_Raporu_Gram_Altin.pdf
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/03betulpehlivan/turkey-gold-price-forecast.git
cd turkey-gold-price-forecast
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

Data files are not included in the repository. Market data is downloaded from Yahoo Finance. Physical gold data requires a free TCMB EVDS API key, kept **only** in an environment variable and never committed:

```powershell
setx EVDS_API_KEY "your-key"      # Windows, applies to new terminals
```

## Usage

```bash
# Data
python -m src.data.collect --start 2005-01-01                       # daily market series
python -m src.data.build_panel --write --processed-dir data/processed/history
python -m src.data.collect_intraday                                 # hourly snapshot (Yahoo keeps ~730 days)
python -m src.data.physical_gold                                    # physical gold series from EVDS

# Research (exploratory, development periods only)
python -m src.models.physical_eval                                  # physical nowcast, 2016-2022
python -m src.models.live_backtest                                  # live protocol replayed over 2012-2022
python -m src.analysis.gain_decomposition                           # overnight gold vs. FX contribution
python -m src.models.interval_study                                 # volatility-scaled intervals

# Live (normally run by the scheduled tasks)
powershell -ExecutionPolicy Bypass -File scripts\sabah_fiziki.ps1
python -m src.live.physical_forecast card                           # live scorecard
python -m src.live.physical_intervals card
```

## Literature Review

A comparative literature review of 23 studies on machine learning-based gold price forecasting, focusing on Turkish gram gold, next-day prediction, model comparison and time-series validation.

📄 **[Read the Literature Review Report](Literatur_Taramasi_Raporu_Gram_Altin.pdf)**

## Roadmap

- [x] Data collection pipeline with immutable snapshots and validation
- [x] Baseline and machine learning models (LR, RF, XGBoost)
- [x] Leakage-free evaluation, statistical tests, pre-registered confirmations
- [x] Probabilistic forecasts (GARCH, GVZ, adaptive conformal calibration)
- [x] Independent tests (`FINAL_EXAM_V1`, `FINAL_EXAM_PHYS_V1`, backward `BACK_EXAM_ONS_V1`)
- [x] Data timing diagnosis and synchronized hourly pipeline
- [x] Physical gold prices in Turkey (EVDS, 1995–2026) and premium analysis
- [x] Live forecasting with pre-registered protocols and automation
- [ ] One-shot evaluation of the live protocols (after 120 forecasts)
- [x] Spot-tracking source diagnosis for the theoretical series (GLD passed the pre-registered rule)
- [ ] `LIVE_V2`: GLD-based theoretical forecast (before the late-November futures roll)
- [ ] Macroeconomic indicators (policy rate, inflation)
- [ ] Weekly and monthly trend forecasts; LSTM benchmark
- [ ] Microsoft SQL Server storage
- [ ] REST API
- [ ] Web interface
- [ ] Final report

## Limitations

- The physical price is a **wholesale** market price; jeweller prices and workmanship are not covered.
- Most of the physical nowcast gain in 2012–2022 depends on an FX timestamp assumption that could only be verified for 2024–2025. The timing-independent part was 5% in development, but it was not confirmed in the 2001–2004 backward independent test. The live record will settle this prospectively.
- Post-2022 physical data was consumed by the independent exam. Models proposed after it (Gap-Rolling, Premium-Anchor) can only be evaluated prospectively.
- Yahoo hourly history covers only about the last two years; the live system archives it daily.
- Daily point forecasts are not claimed to be more accurate than "tomorrow = today" outside the reported conditions.
- **This project is for research and education only and is not financial advice.**

## Contributors

| Name | GitHub | Focus Area |
|------|--------|------------|
| Betül Pehlivan | [@03betulpehlivan](https://github.com/03betulpehlivan) | TBD |
| Enes Karaca | [@EnesKaraca44](https://github.com/EnesKaraca44) | TBD |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
