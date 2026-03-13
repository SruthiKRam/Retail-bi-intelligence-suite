"""
etl_ingestion.py
----------------
Retail BI Intelligence Suite — ETL Ingestion Pipeline

Responsibilities:
  - Extract raw order data from CSV / JSON / SQL source
  - Apply transformation rules (type casting, enrichment, deduplication)
  - Load cleaned data to Parquet (local) or a SQL database (configurable)

Usage:
    python pipelines/etl_ingestion.py --source data/sample_orders.csv --output data/cleaned_orders.parquet
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etl_ingestion.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "order_id", "customer_id", "product_id", "sku",
    "order_date", "quantity", "unit_price", "discount",
    "channel", "region", "category",
]

DTYPE_MAP = {
    "order_id":    str,
    "customer_id": str,
    "product_id":  str,
    "sku":         str,
    "quantity":    int,
    "unit_price":  float,
    "discount":    float,
    "channel":     str,
    "region":      str,
    "category":    str,
}

VALID_CHANNELS = {"organic", "paid_search", "email", "social", "direct", "affiliate"}
VALID_REGIONS  = {"north", "south", "east", "west", "international"}


# ─────────────────────────────────────────────
# Extract
# ─────────────────────────────────────────────
def extract(source_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV or JSON file.

    Parameters
    ----------
    source_path : str
        Path to the raw source file.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed DataFrame.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    logger.info(f"Extracting data from: {source_path}")

    if path.suffix == ".csv":
        df = pd.read_csv(source_path, low_memory=False)
    elif path.suffix == ".json":
        df = pd.read_json(source_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .json")

    logger.info(f"Extracted {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# Validate Schema
# ─────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    """Raise an error if required columns are missing."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed ✓")


# ─────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all transformation steps to the raw DataFrame.

    Steps:
      1. Cast data types
      2. Parse and normalize dates
      3. Remove duplicates
      4. Handle missing values
      5. Standardize categorical columns
      6. Derive computed columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame.
    """
    logger.info("Starting transformation...")
    original_len = len(df)

    # 1. Cast types
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not cast column '{col}' to {dtype}: {e}")

    # 2. Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_year"]  = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"]   = df["order_date"].dt.day_name()
    df["order_week"]  = df["order_date"].dt.isocalendar().week.astype(int)
    logger.info("Date parsing complete ✓")

    # 3. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    dupes_removed = before - len(df)
    if dupes_removed:
        logger.warning(f"Removed {dupes_removed:,} duplicate order_id rows")

    # 4. Handle missing values
    df["discount"] = df["discount"].fillna(0.0)
    df = df.dropna(subset=["order_date", "customer_id", "product_id"])
    logger.info(f"Dropped {original_len - len(df):,} rows with critical nulls")

    # 5. Standardize categoricals
    df["channel"]  = df["channel"].str.strip().str.lower()
    df["region"]   = df["region"].str.strip().str.lower()
    df["category"] = df["category"].str.strip().str.title()
    df["sku"]      = df["sku"].str.strip().str.upper()

    # Flag unknown channels / regions (keep but tag)
    df["channel_valid"] = df["channel"].isin(VALID_CHANNELS)
    df["region_valid"]  = df["region"].isin(VALID_REGIONS)

    unknown_channels = (~df["channel_valid"]).sum()
    if unknown_channels:
        logger.warning(f"{unknown_channels:,} rows have unrecognised channel values")

    # 6. Derived columns
    df["gross_revenue"]    = df["quantity"] * df["unit_price"]
    df["discount_amount"]  = df["gross_revenue"] * df["discount"]
    df["net_revenue"]      = df["gross_revenue"] - df["discount_amount"]
    df["is_discounted"]    = df["discount"] > 0

    logger.info(f"Transformation complete ✓ — {len(df):,} rows retained")
    return df


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
def load(df: pd.DataFrame, output_path: str) -> None:
    """
    Persist the cleaned DataFrame to Parquet or CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data to persist.
    output_path : str
        Target file path (.parquet or .csv).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(output_path, index=False, compression="snappy")
    elif path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")

    size_kb = path.stat().st_size / 1024
    logger.info(f"Loaded {len(df):,} rows → {output_path} ({size_kb:.1f} KB) ✓")


# ─────────────────────────────────────────────
# Pipeline Summary
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the cleaned dataset."""
    print("\n" + "=" * 50)
    print("  ETL PIPELINE SUMMARY")
    print("=" * 50)
    print(f"  Total orders      : {len(df):,}")
    print(f"  Unique customers  : {df['customer_id'].nunique():,}")
    print(f"  Unique SKUs       : {df['sku'].nunique():,}")
    print(f"  Date range        : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"  Gross revenue     : ${df['gross_revenue'].sum():,.2f}")
    print(f"  Net revenue       : ${df['net_revenue'].sum():,.2f}")
    print(f"  Channels          : {sorted(df['channel'].unique())}")
    print(f"  Regions           : {sorted(df['region'].unique())}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run_pipeline(source: str, output: str) -> pd.DataFrame:
    """End-to-end ETL run: extract → validate → transform → load."""
    logger.info("=" * 50)
    logger.info("  RETAIL BI — ETL INGESTION PIPELINE")
    logger.info("=" * 50)

    start = datetime.now()

    df_raw = extract(source)
    validate_schema(df_raw)
    df_clean = transform(df_raw)
    load(df_clean, output)
    print_summary(df_clean)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Pipeline completed in {elapsed:.2f}s ✓")
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail BI — ETL Ingestion Pipeline")
    parser.add_argument(
        "--source", type=str,
        default="data/sample_orders.csv",
        help="Path to raw source file (CSV or JSON)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/cleaned_orders.parquet",
        help="Output path (.parquet or .csv)",
    )
    args = parser.parse_args()

    run_pipeline(source=args.source, output=args.output)
