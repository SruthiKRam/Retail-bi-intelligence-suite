# Architecture Notes

## Data Flow

```
Raw Orders (CSV / DB / JSON)
          │
          ▼
  ┌─────────────────────┐
  │   ETL Ingestion     │  pipelines/etl_ingestion.py
  │   - Extract         │
  │   - Transform       │
  │   - Load → Parquet  │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │   Data Quality      │  pipelines/data_quality.py
  │   - Null checks     │
  │   - Outlier detect  │
  │   - Business rules  │
  │   - Freshness check │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │   KPI Aggregation   │  pipelines/kpi_aggregation.py
  │   - Revenue KPIs    │
  │   - Customer KPIs   │
  │   - Channel KPIs    │
  │   - SKU KPIs        │
  └─────────┬───────────┘
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
  ┌──────┐ ┌────────────┐ ┌────────────┐
  │ MTA  │ │  Product   │ │  Basket    │
  │Shapley│ │  Scoring  │ │  Affinity  │
  └──────┘ └────────────┘ └────────────┘
            │
            ▼
  Power BI / Tableau Dashboards
```

## Tech Decisions

- **Parquet** for columnar storage — fast reads for analytical queries
- **Shapley values** for attribution — theoretically fair, avoids first/last-click bias
- **MinMaxScaler** for SKU scoring — handles different units across dimensions
- **Apriori** for basket affinity — interpretable, threshold-driven itemset mining
