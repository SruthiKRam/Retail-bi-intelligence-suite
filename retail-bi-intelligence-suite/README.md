# 🛒 Retail BI Intelligence Suite

A production-grade Business Intelligence pipeline for retail analytics — covering ETL ingestion, data quality validation, KPI aggregation, multi-touch attribution (Shapley), SKU scoring, and basket affinity analysis.

---

## 📁 Project Structure

```
retail-bi-intelligence-suite/
├── README.md
├── requirements.txt
├── /pipelines
│   ├── etl_ingestion.py       ← ETL: extract, transform, load orders data
│   ├── data_quality.py        ← Validation checks & data profiling
│   └── kpi_aggregation.py     ← KPI calculation logic (revenue, AOV, CLV, etc.)
├── /notebooks
│   ├── mta_shapley_model.ipynb      ← Multi-touch attribution via Shapley values
│   ├── product_scoring.ipynb        ← SKU-level scoring model
│   └── basket_affinity.ipynb        ← Market basket / association rules
├── /data
│   └── sample_orders.csv      ← Synthetic dummy data (no real customer data)
├── /dashboards
│   └── screenshots/           ← Power BI / Tableau dashboard screenshots
└── /docs
    └── architecture_diagram.png
```

---

## 🚀 Features

| Module | Description |
|---|---|
| **ETL Ingestion** | Reads raw CSV/JSON orders, cleans, transforms, and loads to a target (DB or Parquet) |
| **Data Quality** | Schema validation, null checks, outlier detection, and freshness monitoring |
| **KPI Aggregation** | Revenue, AOV, CLV, repeat purchase rate, category contribution |
| **MTA Shapley** | Fair attribution of conversions across marketing touchpoints |
| **Product Scoring** | Composite SKU score: margin × velocity × return rate |
| **Basket Affinity** | Apriori/FP-Growth association rules for cross-sell recommendations |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/retail-bi-intelligence-suite.git
cd retail-bi-intelligence-suite
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
# Run ETL
python pipelines/etl_ingestion.py

# Run data quality checks
python pipelines/data_quality.py

# Generate KPIs
python pipelines/kpi_aggregation.py
```

### 4. Launch notebooks
```bash
jupyter notebook notebooks/
```

---

## 📊 Sample Data

The `/data/sample_orders.csv` file contains **synthetically generated** retail order data. It includes:
- `order_id`, `customer_id`, `product_id`, `sku`
- `order_date`, `quantity`, `unit_price`, `discount`
- `channel`, `region`, `category`

> ⚠️ No real customer data is used anywhere in this project.

---

## 🧱 Architecture

```
Raw Data (CSV/DB)
      ↓
  ETL Ingestion  ──→  Cleaned & Enriched Dataset
      ↓
  Data Quality   ──→  Validation Report
      ↓
  KPI Aggregation ──→  Aggregated KPI Tables
      ↓
  ┌─────────────────────────────┐
  │  Analytics Layer            │
  │  ├── MTA Shapley Model      │
  │  ├── SKU Scoring Model      │
  │  └── Basket Affinity Rules  │
  └─────────────────────────────┘
      ↓
  Power BI / Tableau Dashboards
```

---

## 🛠 Tech Stack

- **Python 3.10+**
- **pandas**, **numpy** — data wrangling
- **scikit-learn** — modeling utilities
- **mlxtend** — association rule mining
- **great_expectations** — data quality
- **matplotlib**, **seaborn** — visualizations
- **jupyter** — interactive notebooks
- **sqlalchemy** — database connectivity (optional)

---

## 📌 Roadmap

- [ ] Add dbt models for warehouse transformations
- [ ] Integrate with Snowflake / BigQuery connector
- [ ] Build automated alerting for KPI anomalies
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Dockerize the ETL pipeline

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
