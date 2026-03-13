"""
data_quality.py
---------------
Retail BI Intelligence Suite — Data Quality & Validation Module

Responsibilities:
  - Schema and column presence checks
  - Null / missing value profiling
  - Statistical outlier detection (IQR & Z-score)
  - Referential integrity checks
  - Data freshness monitoring
  - Generates a human-readable quality report

Usage:
    python pipelines/data_quality.py --input data/cleaned_orders.parquet --report data/quality_report.txt
"""

import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants / Thresholds
# ─────────────────────────────────────────────
NULL_THRESHOLD_PCT   = 5.0     # Flag columns with >5% nulls
OUTLIER_Z_THRESHOLD  = 3.5     # Z-score cutoff for outliers
FRESHNESS_DAYS       = 7       # Data must be newer than N days
MIN_ROWS             = 100     # Warn if dataset has fewer rows

NUMERIC_COLS  = ["quantity", "unit_price", "discount", "gross_revenue", "net_revenue"]
CATEGORY_COLS = ["channel", "region", "category"]
DATE_COLS     = ["order_date"]
ID_COLS       = ["order_id", "customer_id", "product_id"]

EXPECTED_CHANNELS  = {"organic", "paid_search", "email", "social", "direct", "affiliate"}
EXPECTED_REGIONS   = {"north", "south", "east", "west", "international"}

BUSINESS_RULES = {
    "unit_price_min": 0.01,
    "unit_price_max": 100_000,
    "quantity_min":   1,
    "quantity_max":   10_000,
    "discount_min":   0.0,
    "discount_max":   1.0,
}


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_data(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif path.suffix == ".csv":
        df = pd.read_csv(input_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    logger.info(f"Loaded {len(df):,} rows from {input_path}")
    return df


# ─────────────────────────────────────────────
# Check 1: Row Count
# ─────────────────────────────────────────────
def check_row_count(df: pd.DataFrame) -> dict:
    n = len(df)
    status = "PASS" if n >= MIN_ROWS else "WARN"
    return {
        "check": "Row Count",
        "status": status,
        "detail": f"{n:,} rows (minimum expected: {MIN_ROWS:,})",
    }


# ─────────────────────────────────────────────
# Check 2: Schema Presence
# ─────────────────────────────────────────────
def check_schema(df: pd.DataFrame) -> dict:
    required = set(ID_COLS + NUMERIC_COLS + CATEGORY_COLS + DATE_COLS)
    missing  = required - set(df.columns)
    status   = "PASS" if not missing else "FAIL"
    return {
        "check": "Schema Presence",
        "status": status,
        "detail": f"Missing columns: {missing}" if missing else "All required columns present",
    }


# ─────────────────────────────────────────────
# Check 3: Null Profiling
# ─────────────────────────────────────────────
def check_nulls(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        status = "PASS" if null_pct <= NULL_THRESHOLD_PCT else "WARN"
        if null_pct > 0:
            results.append({
                "check": f"Null Check — {col}",
                "status": status,
                "detail": f"{null_pct:.2f}% null ({df[col].isna().sum():,} rows)",
            })
    if not results:
        results.append({"check": "Null Check", "status": "PASS", "detail": "No nulls found"})
    return results


# ─────────────────────────────────────────────
# Check 4: Duplicate IDs
# ─────────────────────────────────────────────
def check_duplicates(df: pd.DataFrame) -> dict:
    if "order_id" not in df.columns:
        return {"check": "Duplicate order_id", "status": "SKIP", "detail": "Column not found"}
    dupes = df["order_id"].duplicated().sum()
    status = "PASS" if dupes == 0 else "FAIL"
    return {
        "check": "Duplicate order_id",
        "status": status,
        "detail": f"{dupes:,} duplicate order IDs found",
    }


# ─────────────────────────────────────────────
# Check 5: Outlier Detection
# ─────────────────────────────────────────────
def check_outliers(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()

        # Z-score method
        z_scores = np.abs(stats.zscore(series))
        n_outliers = (z_scores > OUTLIER_Z_THRESHOLD).sum()

        # IQR method
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_iqr = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()

        status = "PASS" if n_outliers == 0 else "WARN"
        results.append({
            "check": f"Outlier Detection — {col}",
            "status": status,
            "detail": (
                f"Z-score outliers: {n_outliers:,} | IQR outliers: {n_iqr:,} | "
                f"min={series.min():.2f} | max={series.max():.2f} | mean={series.mean():.2f}"
            ),
        })
    return results


# ─────────────────────────────────────────────
# Check 6: Business Rules
# ─────────────────────────────────────────────
def check_business_rules(df: pd.DataFrame) -> list[dict]:
    results = []

    checks = [
        ("unit_price", BUSINESS_RULES["unit_price_min"], BUSINESS_RULES["unit_price_max"]),
        ("quantity",   BUSINESS_RULES["quantity_min"],   BUSINESS_RULES["quantity_max"]),
        ("discount",   BUSINESS_RULES["discount_min"],   BUSINESS_RULES["discount_max"]),
    ]

    for col, min_val, max_val in checks:
        if col not in df.columns:
            continue
        violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
        status = "PASS" if violations == 0 else "FAIL"
        results.append({
            "check": f"Business Rule — {col} in [{min_val}, {max_val}]",
            "status": status,
            "detail": f"{violations:,} violations",
        })

    return results


# ─────────────────────────────────────────────
# Check 7: Categorical Integrity
# ─────────────────────────────────────────────
def check_categoricals(df: pd.DataFrame) -> list[dict]:
    results = []

    mapping = {
        "channel": EXPECTED_CHANNELS,
        "region":  EXPECTED_REGIONS,
    }

    for col, valid_set in mapping.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - valid_set
        status = "PASS" if not unexpected else "WARN"
        results.append({
            "check": f"Categorical Integrity — {col}",
            "status": status,
            "detail": f"Unexpected values: {unexpected}" if unexpected else "All values within expected set",
        })

    return results


# ─────────────────────────────────────────────
# Check 8: Data Freshness
# ─────────────────────────────────────────────
def check_freshness(df: pd.DataFrame) -> dict:
    if "order_date" not in df.columns:
        return {"check": "Data Freshness", "status": "SKIP", "detail": "order_date not found"}

    max_date = pd.to_datetime(df["order_date"]).max()
    cutoff   = datetime.now() - timedelta(days=FRESHNESS_DAYS)
    status   = "PASS" if max_date >= cutoff else "WARN"

    return {
        "check": "Data Freshness",
        "status": status,
        "detail": f"Most recent order date: {max_date.date()} (threshold: within {FRESHNESS_DAYS} days)",
    }


# ─────────────────────────────────────────────
# Aggregate & Report
# ─────────────────────────────────────────────
def run_all_checks(df: pd.DataFrame) -> list[dict]:
    all_results = []

    all_results.append(check_row_count(df))
    all_results.append(check_schema(df))
    all_results.extend(check_nulls(df))
    all_results.append(check_duplicates(df))
    all_results.extend(check_outliers(df))
    all_results.extend(check_business_rules(df))
    all_results.extend(check_categoricals(df))
    all_results.append(check_freshness(df))

    return all_results


def generate_report(results: list[dict], output_path: str | None = None) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("  RETAIL BI — DATA QUALITY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "⏭️ "}.get(status, "?")
        lines.append(f"\n{icon} [{status}] {r['check']}")
        lines.append(f"   └─ {r['detail']}")

    lines.append("\n" + "-" * 65)
    lines.append(f"  SUMMARY  |  PASS: {counts['PASS']}  |  WARN: {counts['WARN']}  |  FAIL: {counts['FAIL']}  |  SKIP: {counts['SKIP']}")
    lines.append("=" * 65 + "\n")

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Quality report saved → {output_path}")

    return report


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail BI — Data Quality Checks")
    parser.add_argument("--input",  type=str, default="data/cleaned_orders.parquet")
    parser.add_argument("--report", type=str, default="data/quality_report.txt")
    args = parser.parse_args()

    df      = load_data(args.input)
    results = run_all_checks(df)
    generate_report(results, output_path=args.report)
