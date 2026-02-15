# RetailChain Big Data Analytics Pipeline

A PySpark-based end-to-end analytics pipeline for supply chain data. Ingest data from HDFS, clean and transform it, run 10+ analytical modules, and train an ML model for late delivery risk prediction.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Hadoop](https://img.shields.io/badge/Hadoop-HDFS-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## Features

- **Data ingestion** — Load CSV from HDFS with schema inference
- **Data cleaning** — Parse dates, cast numeric types, handle encoding
- **Curated storage** — Write cleaned data to Parquet format
- **10 analytics modules:**
  1. Monthly sales & profit trends
  2. Category contribution analysis
  3. Shipping mode performance
  4. Regional performance
  5. Customer segment analysis
  6. Time-based patterns (hour, day of week)
  7. Product performance (top/bottom)
  8. Order status distribution
  9. Summary statistics
  10. Correlation analysis
- **ML model** — Logistic Regression for late delivery risk prediction
- **Visualizations** — Auto-generated PNG charts for all analyses

## Tech Stack

| Component | Version |
|-----------|---------|
| PySpark   | 3.5.0   |
| Hadoop    | 3.2.1   |
| Matplotlib| ≥3.5.0  |
| Python    | 3.x     |
| Docker    | Compose 3.8 |

## Project Structure

```
bigdata-assignment/
├── analytics_pipeline.py    # Main pipeline script
├── data/
│   ├── analytics_pipeline.py # Pipeline copy (for Jupyter/hosted env)
│   └── DataCoSupplyChainDataset.csv
├── outputs/                 # Generated charts (created at runtime)
├── docker-compose.yml       # Hadoop + Jupyter Spark setup
├── requirements.txt
└── README.md
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.x (for local runs)
- 4GB+ RAM recommended

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bigdata-assignment.git
cd bigdata-assignment
```

### 2. Start the infrastructure

```bash
docker compose up -d
```

This starts:

- **HDFS NameNode** (port 9870, 8020)
- **HDFS DataNode**
- **Jupyter PySpark Notebook** (port 8888)

### 3. Upload data to HDFS

The pipeline expects the raw CSV at:

```
hdfs://namenode:8020/data/retailchain/raw/DataCoSupplyChainDataset.csv
```

From your host machine:

```bash
# Create raw directory in HDFS
docker exec -it namenode hdfs dfs -mkdir -p /data/retailchain/raw

# Copy dataset into HDFS (data is in data/ folder)
docker exec -i namenode hdfs dfs -put -f /Workspace/DataCoSupplyChainDataset.csv /data/retailchain/raw/
```

If `DataCoSupplyChainDataset.csv` lives elsewhere:

```bash
docker cp /path/to/DataCoSupplyChainDataset.csv namenode:/tmp/
docker exec namenode hdfs dfs -put -f /tmp/DataCoSupplyChainDataset.csv /data/retailchain/raw/
```

### 4. Run the pipeline

**Option A — Jupyter Notebook (recommended)**

1. Open Jupyter: [http://localhost:8888](http://localhost:8888)
2. Copy your token from the terminal logs
3. Open `data/analytics_pipeline.py` or create a new notebook
4. Run the script:
   ```python
   %run /home/jovyan/work/analytics_pipeline.py
   ```

**Option B — Command line (inside Spark container)**

```bash
docker exec -it spark-notebook bash
spark-submit /home/jovyan/work/analytics_pipeline.py
```

### 5. View outputs

Charts are written to `outputs/` (mounted from `./outputs`):

- `monthly_trends.png`
- `category_analysis.png`
- `shipping_analysis.png`
- `regional_analysis.png`
- `customer_segment_analysis.png`
- `time_analysis.png`
- `product_performance.png`
- `order_status_analysis.png`
- `summary_statistics.png`
- `correlation_analysis.png`

## Dataset

Uses the **DataCo Supply Chain Dataset**. Ensure your CSV includes columns such as:

- `order date (DateOrders)`, `shipping date (DateOrders)`
- `Sales`, `Order Profit Per Order`, `Late_delivery_risk`
- `Days for shipping (real)`, `Days for shipment (scheduled)`
- `Shipping Mode`, `Category Name`, `Order Region`, `Product Name`, `Customer Segment`, `Order Status`

## HDFS Web UI

- **NameNode UI:** [http://localhost:9870](http://localhost:9870) — browse HDFS and check data layout

## License

MIT (or your preferred license)
