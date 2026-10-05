# Turkey Gold Price Forecast

Machine learning-based analysis and forecasting system for physical gold prices in Turkey, using historical prices, the USD/TRY exchange rate, ounce gold, and macroeconomic indicators.

> **Status:** Work in progress. This is a graduation (capstone) project.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

Physical gold is one of the most widely held savings instruments in Turkey, and its price is shaped by global ounce gold prices, the USD/TRY exchange rate, and domestic economic conditions. This project collects and analyzes these data sources and builds machine learning models that produce **short-term price forecasts** and **trend predictions** for physical gold in the Turkish market.

The system is designed as an end-to-end pipeline: data collection, storage, preprocessing, model training, evaluation, and presentation through a REST API and a web interface.

## Features

- Collection of historical physical gold prices in Turkey
- Integration of USD/TRY exchange rates, ounce gold prices, and economic indicators
- Exploratory data analysis and data visualization
- Feature engineering for time series data
- Short-term price forecasting with machine learning models (XGBoost, LSTM)
- Trend prediction and model performance comparison
- Data storage in Microsoft SQL Server
- REST API for serving predictions
- Web interface for interactive visualization

## Tech Stack

| Area | Technologies |
|------|--------------|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, LSTM |
| Visualization | Matplotlib / Plotly (TBD) |
| Backend | REST API (framework TBD) |
| Frontend | Web interface (TBD) |
| Database | Microsoft SQL Server |

## Project Structure

```
turkey-gold-price-forecast/
├── data/
│   ├── raw/              # Raw collected data (not tracked by git)
│   └── processed/        # Cleaned and feature-engineered data
├── notebooks/            # Exploratory analysis and experiments
├── src/
│   ├── data/             # Data collection and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── api/              # REST API
│   └── visualization/    # Plotting utilities
├── web/                  # Web interface
├── tests/                # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

> The structure may evolve as the project develops.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Microsoft SQL Server (for the database component)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/03betulpehlivan/turkey-gold-price-forecast.git
cd turkey-gold-price-forecast

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root for sensitive settings (never commit this file):

```
DB_SERVER=your_server
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

## Usage

> Usage instructions will be added as the modules are implemented.

```bash
# Example (planned)
python -m src.models.train
python -m src.api.main
```

## Methodology

1. **Data collection:** Gather historical physical gold prices, USD/TRY rates, ounce gold prices, and economic indicators.
2. **Preprocessing:** Clean, align, and normalize the time series data.
3. **Feature engineering:** Create lag features, moving averages, returns, and volatility measures.
4. **Modeling:** Train and compare XGBoost and LSTM models for short-term forecasting.
5. **Evaluation:** Assess models with metrics such as MAE, RMSE, and MAPE using time-series-aware validation.
6. **Deployment:** Serve predictions through a REST API and a web interface.

## Results

> Model performance metrics and forecast charts will be added here.

## Roadmap

- [ ] Data collection pipeline
- [ ] Database schema and storage
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline and advanced models
- [ ] Model evaluation and comparison
- [ ] REST API
- [ ] Web interface
- [ ] Documentation and final report

## Contributing

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear messages.
3. Push your branch and open a Pull Request.
4. Wait for a review before merging into `main`.

Always run `git pull` on `main` before starting new work.

## Contributors

- [Betül Pehlivan](https://github.com/03betulpehlivan)
- Your teammate's name (GitHub profile link)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
