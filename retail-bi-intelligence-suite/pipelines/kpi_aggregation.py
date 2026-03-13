"""
kpi_aggregation.py
------------------
Retail BI Intelligence Suite — KPI Aggregation Module

Computes the following KPIs from cleaned order data:

  Revenue KPIs:
    - Gross Revenue, Net Revenue, Discount Rate
    - Revenue by Channel, Region, Category
    - Month-over-Month (MoM) revenue growth

  Customer KPIs:
    - Total Unique Customers
    - Average Order Value (AOV)
    - Customer Lifetime Value (CLV) estimate
    - Repeat Purchase Rate
    - New vs Returning customer split

  Product KPIs:
    - Units Sold, Revenue per SKU
    - Top 10 SKUs by revenue

  Operational KPIs:
    - Orders per Day
    - Average basket size (items per order)

Usage:
    python pipelines/kpi_aggregation.py --input data/cleaned_orders.parquet --output data/kpis.csv
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_parquet(path) if p.suffix == ".parquet" else pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    logger.info(f"Loaded {len(df):,} rows for KPI computation")
    return df


# ─────────────────────────────────────────────
# Revenue KPIs
# ─────────────────────────────────────────────
def compute_revenue_kpis(df: pd.DataFrame) -> dict:
    logger.info("Computing revenue KPIs...")
    total_gross    = df["gross_revenue"].sum()
    total_net      = df["net_revenue"].sum()
    total_discount = df["discount_amount"].sum()
    discount_rate  = (total_discount / total_gross * 100) if total_gross > 0 else 0

    # MoM growth — last complete month vs previous
    df["year_month"] = df["order_date"].dt.to_period("M")
    monthly = df.groupby("year_month")["net_revenue"].sum().sort_index()

    mom_growth = None
    if len(monthly) >= 2:
        last_month = monthly.iloc[-1]
        prev_month = monthly.iloc[-2]
        mom_growth = ((last_month - prev_month) / prev_month * 100) if prev_month != 0 else None

    return {
        "total_gross_revenue":    round(total_gross, 2),
        "total_net_revenue":      round(total_net, 2),
        "total_discount_amount":  round(total_discount, 2),
        "overall_discount_rate":  round(discount_rate, 2),
        "mom_revenue_growth_pct": round(mom_growth, 2) if mom_growth is not None else None,
    }


# ─────────────────────────────────────────────
# Customer KPIs
# ─────────────────────────────────────────────
def compute_customer_kpis(df: pd.DataFrame) -> dict:
    logger.info("Computing customer KPIs...")

    total_orders    = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    total_revenue   = df["net_revenue"].sum()

    # AOV
    order_revenue = df.groupby("order_id")["net_revenue"].sum()
    aov = order_revenue.mean()

    # Orders per customer
    orders_per_customer = df.groupby("customer_id")["order_id"].nunique()

    # Repeat purchase rate
    repeat_buyers    = (orders_per_customer > 1).sum()
    repeat_rate      = repeat_buyers / total_customers * 100 if total_customers > 0 else 0

    # CLV estimate (average revenue × average order frequency)
    avg_revenue_per_customer = df.groupby("customer_id")["net_revenue"].sum().mean()
    avg_orders_per_customer  = orders_per_customer.mean()
    clv_estimate = avg_revenue_per_customer  # simplified; extend with churn rate if available

    return {
        "total_customers":           total_customers,
        "total_orders":              total_orders,
        "average_order_value":       round(aov, 2),
        "repeat_purchase_rate_pct":  round(repeat_rate, 2),
        "avg_orders_per_customer":   round(avg_orders_per_customer, 2),
        "clv_estimate":              round(clv_estimate, 2),
    }


# ─────────────────────────────────────────────
# Channel KPIs
# ─────────────────────────────────────────────
def compute_channel_kpis(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing channel KPIs...")
    return (
        df.groupby("channel")
        .agg(
            orders       =("order_id",    "nunique"),
            customers    =("customer_id", "nunique"),
            gross_revenue=("gross_revenue","sum"),
            net_revenue  =("net_revenue",  "sum"),
            discount_rate=("discount",     "mean"),
            aov          =("net_revenue",  lambda x: x.sum() / df.loc[x.index, "order_id"].nunique()),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
        .round(2)
    )


# ─────────────────────────────────────────────
# Category KPIs
# ─────────────────────────────────────────────
def compute_category_kpis(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing category KPIs...")
    cat = (
        df.groupby("category")
        .agg(
            orders       =("order_id",     "nunique"),
            units_sold   =("quantity",     "sum"),
            gross_revenue=("gross_revenue","sum"),
            net_revenue  =("net_revenue",  "sum"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
    )
    cat["revenue_share_pct"] = (cat["net_revenue"] / cat["net_revenue"].sum() * 100).round(2)
    return cat.round(2)


# ─────────────────────────────────────────────
# SKU / Product KPIs
# ─────────────────────────────────────────────
def compute_sku_kpis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info(f"Computing top {top_n} SKU KPIs...")
    sku = (
        df.groupby(["sku", "product_id"])
        .agg(
            orders      =("order_id",     "nunique"),
            units_sold  =("quantity",     "sum"),
            net_revenue =("net_revenue",  "sum"),
            avg_price   =("unit_price",   "mean"),
            avg_discount=("discount",     "mean"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
        .head(top_n)
        .round(2)
    )
    return sku


# ─────────────────────────────────────────────
# Regional KPIs
# ─────────────────────────────────────────────
def compute_region_kpis(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing regional KPIs...")
    return (
        df.groupby("region")
        .agg(
            orders      =("order_id",     "nunique"),
            customers   =("customer_id",  "nunique"),
            net_revenue =("net_revenue",  "sum"),
            units_sold  =("quantity",     "sum"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
        .round(2)
    )


# ─────────────────────────────────────────────
# Monthly Trend KPIs
# ─────────────────────────────────────────────
def compute_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing monthly trend...")
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    trend = (
        df.groupby("year_month")
        .agg(
            orders      =("order_id",     "nunique"),
            customers   =("customer_id",  "nunique"),
            net_revenue =("net_revenue",  "sum"),
            units_sold  =("quantity",     "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    trend["revenue_mom_growth"] = trend["net_revenue"].pct_change() * 100
    return trend.round(2)


# ─────────────────────────────────────────────
# Print KPI Dashboard
# ─────────────────────────────────────────────
def print_kpi_dashboard(revenue: dict, customer: dict) -> None:
    print("\n" + "=" * 60)
    print("   RETAIL BI — KPI DASHBOARD")
    print("=" * 60)

    print("\n📦 REVENUE KPIs")
    for k, v in revenue.items():
        label = k.replace("_", " ").title()
        val = f"${v:,.2f}" if "revenue" in k or "amount" in k or "clv" in k else (
              f"{v:.2f}%" if "pct" in k or "rate" in k or "growth" in k else str(v))
        print(f"  {label:<35}: {val}")

    print("\n👤 CUSTOMER KPIs")
    for k, v in customer.items():
        label = k.replace("_", " ").title()
        val = f"${v:,.2f}" if "value" in k or "aov" in k or "clv" in k else (
              f"{v:.2f}%" if "pct" in k or "rate" in k else f"{v:,}")
        print(f"  {label:<35}: {val}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# Save Outputs
# ─────────────────────────────────────────────
def save_kpis(kpi_dict: dict, dataframes: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save summary KPIs as CSV
    summary = pd.DataFrame([{**kpi_dict["revenue"], **kpi_dict["customer"]}])
    summary.to_csv(out / "kpi_summary.csv", index=False)
    logger.info(f"Saved kpi_summary.csv → {out}")

    # Save each dimensional breakdown
    for name, df_kpi in dataframes.items():
        path = out / f"kpi_{name}.csv"
        df_kpi.to_csv(path, index=False)
        logger.info(f"Saved {path.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run_kpi_pipeline(input_path: str, output_dir: str) -> None:
    logger.info("=" * 60)
    logger.info("  RETAIL BI — KPI AGGREGATION PIPELINE")
    logger.info("=" * 60)

    df = load_data(input_path)

    revenue_kpis  = compute_revenue_kpis(df)
    customer_kpis = compute_customer_kpis(df)

    channel_kpis  = compute_channel_kpis(df)
    category_kpis = compute_category_kpis(df)
    sku_kpis      = compute_sku_kpis(df, top_n=20)
    region_kpis   = compute_region_kpis(df)
    monthly_trend = compute_monthly_trend(df)

    print_kpi_dashboard(revenue_kpis, customer_kpis)

    save_kpis(
        kpi_dict={"revenue": revenue_kpis, "customer": customer_kpis},
        dataframes={
            "channel":       channel_kpis,
            "category":      category_kpis,
            "top_skus":      sku_kpis,
            "region":        region_kpis,
            "monthly_trend": monthly_trend,
        },
        output_dir=output_dir,
    )
    logger.info("KPI aggregation complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail BI — KPI Aggregation Pipeline")
    parser.add_argument("--input",  type=str, default="data/cleaned_orders.parquet")
    parser.add_argument("--output", type=str, default="data/kpi_outputs/")
    args = parser.parse_args()

    run_kpi_pipeline(input_path=args.input, output_dir=args.output)
