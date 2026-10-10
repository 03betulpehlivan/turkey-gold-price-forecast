
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/theoretical_gold.csv")


def validate_data():
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("=== DATA VALIDATION REPORT ===")
    print(f"Row count: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} - {df['Date'].max().date()}")
    print(f"Duplicate dates: {df['Date'].duplicated().sum()}")
    print(f"Missing values: {df.isna().sum().sum()}")

    price_columns = ["gold_usd_oz", "usdtry", "theoretical_gram_try"]

    for column in price_columns:
        print(f"{column} - zero or negative values: {(df[column] <= 0).sum()}")

    print("\nData validation completed.")


if __name__ == "__main__":
    validate_data()

