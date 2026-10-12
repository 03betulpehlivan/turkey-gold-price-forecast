# Turkey Gold Price Forecast

Machine learning-based forecasting and analysis of gram gold prices in Turkey, built on ounce gold, USD/TRY and market indicators, with a strong emphasis on **leakage-free, independently tested evaluation**.

> **Status:** Work in progress (graduation project, Selçuk University, Computer Engineering).
> The research and data pipeline are well advanced; the database, REST API and web interface are planned.

> **Türkçe özet:** Bu proje, Türkiye'de gram altın fiyatı için makine öğrenmesi tabanlı tahmin ve analiz sistemi geliştirir. Ons altın, USD/TRY ve piyasa göstergeleri kullanılır. Modeller kronolojik, sızıntısız testlerle ve önceden kayıtlı bağımsız sınavlarla değerlendirilir. Temel bulgular: (1) Günlük nokta tahmininde karmaşık modeller "yarın = bugün" referansından anlamlı biçimde daha isabetli değildir; bu, 2026 bağımsız testinde de doğrulanmıştır. (2) Fiyatın hangi aralıkta olacağına dair olasılıklı tahminler, oynaklık modelleri ve uyarlanabilir kalibrasyon ile ölçülebilir biçimde iyileştirilmiştir. (3) Yaygın kullanılan veri kaynağında ons ve kur fiyatlarının ~13–16 saat farklı anlara ait olduğu tespit edilmiş ve eşzamanlı bir veri hattı kurulmuştur.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [What the System Produces](#what-the-system-produces)
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

Gram gold is one of the most widely held savings instruments in Turkey. Its price is driven mainly by the global ounce gold price and the USD/TRY exchange rate. This project:

1. collects and validates daily and hourly market data (gold, USD/TRY and 9 additional market series),
2. builds a theoretical gram gold price, `ounce (USD) × USD/TRY / 31.1035`,
3. compares baseline, statistical and machine learning models for next-day **point forecasts**,
4. develops calibrated **probabilistic forecasts** (80% and 95% prediction intervals),
5. evaluates everything with chronological walk-forward validation, pre-registered decision rules and one-shot independent tests.

The current price definition is the **theoretical** gram price, not a jeweller's buy/sell price. Physical market prices and the jeweller premium are on the roadmap.

## Key Findings

All figures below come from real-data runs; experiment records are in `research/` and `data/processed/results/` (the latter is not tracked by git).

### 1. Next-day point forecasts: no model beats the simple reference

| Evaluation | Forecasts | "Tomorrow = today" | Best/selected model | Result |
|---|---:|---|---|---|
| Development, 2018-01 – 2023-10 (exploratory) | 1,461 | MAE 6.06 TL, MAPE 1.07% | Best model within ~1% of the reference | No model significantly better (Holm-adjusted DM, block bootstrap) |
| **Independent 2026 test (FINAL_EXAM_V1)** | 194 | MAE 90.12 TL, MAPE 1.347% | Mean-Return: MAE 89.93 TL, MAPE 1.345% | **Practically equivalent**: difference −0.21% [95% CI −1.34%, +1.17%] |

Linear Regression, Random Forest, XGBoost and factor-augmented models (dollar index, US yields, VIX, S&P 500, silver, Brent, BIST 100, EUR/TRY) were tested under the same protocol. Errors in TL scale with the price level; the percentage error (~1–1.35%) reflects daily market volatility rather than model choice.

### 2. Probabilistic forecasts: volatility is predictable

- Volatility clustering is strong (Ljung-Box on |returns|, p ≈ 10⁻¹⁶³).
- Development period: GARCH-based intervals reduced pinball loss by ~10% versus constant volatility and removed the clustering of interval violations.
- Combining option-implied gold volatility (CBOE GVZ, lagged one day) with **adaptive conformal calibration (ACI)** raised the worst-year 95% coverage from 0.888 (GARCH, 2018) to 0.920.
- **Pre-registered confirmation (2023-10 – 2025-12, 547 forecasts):** 80% / 95% coverage 0.788 / 0.940; all four pre-registered criteria were met.
- **Independent 2026 test:** coverage 0.794 / 0.938 with correct unconditional calibration (Kupiec p = 0.83 / 0.46), but violations of the 80% interval **clustered** (Christoffersen p ≈ 0.05). The pre-registered interval hypothesis was therefore **not supported** in 2026.

### 3. Data timing problem found and addressed

Diagnostics on hourly data showed that in the commonly used daily source the gold close corresponds to ~18:00 UTC (COMEX settlement), while the USD/TRY close corresponds to ~02:00–05:00 UTC. The usual "theoretical gram price" therefore multiplies prices taken **13–16 hours apart**. A synchronized series was built in which ounce gold and USD/TRY come from the **same hourly bar** (14:00 New York), with an archive of immutable hourly snapshots.

An open issue: the continuous gold futures series rolls contracts on different days in the hourly and daily data, creating artificial jumps of ~1.5–1.9%. Spot-tracking sources (spot XAU/USD, gold ETFs) are being evaluated.

## What the System Produces

For each day, the live module (`src/live/forecast.py`) records:

- the synchronized theoretical gram price and the exact timestamp of the underlying bar,
- a next-day point forecast (Mean-Return) and the "tomorrow = today" reference,
- 80% and 95% prediction intervals,
- after the outcome is known: the TL error and whether each interval contained the realized price.

Forecast records are append-only, integrity-checked (per-row hash) and committed to git. Live forecasting is **paused** until the futures roll issue above is resolved.

## Methodology

- **Chronological walk-forward evaluation:** expanding window, retraining every 63 trading days, 1-row purge; no random splits.
- **Leakage controls:** truncation-invariance tests, target and price alignment checks on real data, explicit timestamps for every input.
- **Pre-registration:** hypotheses, decision rules, code and data hashes are written before each confirmatory run. Confirmatory runs execute once; repeated attempts are refused and logged in `research/registry.jsonl`.
- **Statistics:** Diebold-Mariano with Newey-West variance and HLN small-sample correction, Holm correction for multiple comparisons, paired moving-block bootstrap confidence intervals, Kupiec and Christoffersen tests for interval calibration.
- **Software quality:** 332 unit tests on synthetic data, including mutation checks to verify that the tests detect deliberate errors.

## Tech Stack

| Area | Technologies |
|------|--------------|
| Language | Python 3.11 |
| Data | pandas, NumPy, yfinance |
| Modelling | scikit-learn (Linear Regression, Random Forest), XGBoost, arch (GARCH), SciPy |
| Visualization | Matplotlib |
| Testing | unittest |
| Planned | Microsoft SQL Server, REST API (FastAPI), web interface, LSTM benchmark |

## Project Structure

```
turkey-gold-price-forecast/
├── data/                  # raw snapshots and processed data (not tracked by git)
├── docs/                  # data platform notes, status notes
├── research/              # pre-registrations, registry, independent test results, live log
├── scripts/
│   └── gunluk_tahmin.ps1  # daily pipeline (archive -> series -> scoring -> forecast)
├── src/
│   ├── data/              # collection, validation, dataset/panel building, synchronized series
│   ├── analysis/          # diagnostics, timing and data-source analyses
│   ├── models/            # point, interval, factor, implied-volatility, ACI models; final exam
│   └── live/              # live forecasting and scoring
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

Data files are not included in the repository; the collection commands below download them from Yahoo Finance.

## Usage

```bash
# Data
python -m src.data.collect                 # daily series (11 series incl. GVZ), immutable snapshot
python -m src.data.build_panel --write     # aligned market panel
python -m src.data.collect_intraday        # hourly snapshot (Yahoo keeps ~730 days)
python -m src.data.sync_dataset            # synchronized daily series (14:00 New York)

# Models and evaluation (development period)
python -m src.models.train                 # baseline and ML models
python -m src.models.point_eval            # point forecasts in TL, vs. "tomorrow = today"
python -m src.models.intervals             # GARCH-based prediction intervals
python -m src.models.adaptive              # adaptive conformal calibration

# Diagnostics
python -m src.analysis.fx_close_time       # which hour each daily close belongs to
python -m src.analysis.spot_probe          # spot-tracking gold source candidates

# Daily pipeline (Windows)
powershell -ExecutionPolicy Bypass -File scripts\gunluk_tahmin.ps1 -TahminYok
```

## Literature Review

A comparative literature review of 23 studies on machine learning-based gold price forecasting, focusing on Turkish gram gold, next-day prediction, model comparison and time-series validation.

📄 **[Read the Literature Review Report](Literatur_Taramasi_Raporu_Gram_Altin.pdf)**

## Roadmap

- [x] Data collection pipeline with immutable snapshots and validation
- [x] Baseline and machine learning models (LR, RF, XGBoost)
- [x] Leakage-free evaluation, statistical tests, pre-registered confirmations
- [x] Probabilistic forecasts (GARCH, GVZ, adaptive conformal calibration)
- [x] Independent 2026 test (FINAL_EXAM_V1)
- [x] Data timing diagnosis and synchronized hourly pipeline
- [ ] Spot-tracking gold source (resolve futures roll artifacts)
- [ ] Live forecasting with public track record
- [ ] Physical gold prices in Turkey and jeweller premium analysis
- [ ] Macroeconomic indicators (policy rate, inflation)
- [ ] Weekly and monthly trend forecasts; LSTM benchmark
- [ ] Microsoft SQL Server storage
- [ ] REST API
- [ ] Web interface
- [ ] Final report

## Limitations

- Prices are **theoretical** (ounce × USD/TRY); they are not jeweller or Grand Bazaar prices.
- Data comes from Yahoo Finance; hourly history is limited to roughly the last two years.
- Daily point forecasts are not more accurate than "tomorrow = today"; the project does not claim otherwise.
- **This project is for research and education only and is not financial advice.**

## Contributors

| Name | GitHub | Focus Area |
|------|--------|------------|
| Betül Pehlivan | [@03betulpehlivan](https://github.com/03betulpehlivan) | TBD |
| Enes Karaca | [@EnesKaraca44](https://github.com/EnesKaraca44) | TBD |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
