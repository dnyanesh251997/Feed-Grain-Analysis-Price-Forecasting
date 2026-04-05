# 🌾 Feed Grain Analysis & Price Forecasting

A big data pipeline and machine learning project that ingests USDA feed grain data from Azure Data Lake Storage, processes it using Apache Spark (PySpark), and produces three analytical outputs — price forecasts, demand analysis, and production trend classification — stored in PostgreSQL.

---

## 📁 Project Structure

```
FeedGrain/
├── FeedGrains.csv          # Raw USDA feed grain dataset (~523K rows)
├── Feedgrain.py            # Main PySpark ML pipeline script
├── feedgrain.bat           # Windows batch launcher
├── Code.ipynb              # Jupyter Notebook (exploratory / development version)
├── demand.csv              # Output: Production vs Demand predictions
├── trends.csv              # Output: Production trend classification
└── xgboost.csv             # Output: Forecasted grain prices (2026)
```

---

## 🎯 Research Questions Answered

### Q1 — Price Forecasting (XGBoost)
> *How can historical trends be leveraged to accurately forecast future feed grain prices using machine learning models?*

An **XGBoost Regressor** is trained on historical annual price data for Corn, Oats, Barley, and Sorghum. Seasonal signals (sin/cos month encoding) are added and the model produces **monthly price forecasts for 2026**.

**Output table:** `xgboost` in PostgreSQL | `xgboost.csv`

| Metric | Value |
|--------|-------|
| RMSE   | Reported at runtime |
| MAPE   | Reported at runtime |
| R²     | Reported at runtime |

---

### Q2 — Demand vs Production (Random Forest Regression)
> *How has the market demand for major feed grains evolved in relation to their production over the years?*

A **PySpark Random Forest Regressor** models demand as a function of production, year, and commodity type. Supply/use attributes are classified into "Production" vs "Demand" categories and grouped by year.

**Output table:** `demand` in PostgreSQL | `demand.csv`

| Metric | Value |
|--------|-------|
| R²     | Reported at runtime |

---

### Q3 — Production Trend Classification (Random Forest Classifier)
> *What are the patterns and trends in feed grain production over different seasons or years?*

A **PySpark Random Forest Classifier** labels each production record as `Low`, `Medium`, or `High` using IQR-based thresholds. Features include grain type, region, season (Timeperiod_Desc), and year.

**Output table:** `Trends` in PostgreSQL | `trends.csv`

| Metric   | Value |
|----------|-------|
| Accuracy | Reported at runtime |
| F1 Score | Reported at runtime |

---
## 🔄 Methodology Flow

The pipeline follows a structured end-to-end architecture:

```
Data Selection (USDA Feedgrain Dataset)
        │
        ▼
Azure Blob Storage (Secure Cloud Storage)
        │
        ▼
Data Ingestion via PySpark (Apache Spark Session)
        │
        ▼
ETL Processing
├── Cleaning (whitespace, duplicates, nulls, outliers via IQR)
├── Transformation (schema casting, feature engineering)
└── Feature Engineering (cyclical sin/cos month, commodity index, production quantiles)
        │
        ▼
┌───────────────────────────────────────────────┐
│         Machine Learning Models               │
├───────────────┬───────────────┬───────────────┤
│   XGBoost     │  Random Forest│  Random Forest│
│   Regressor   │  Regressor    │  Classifier   │
│ (Price Pred.) │ (Demand Pred.)│ (Prod. Level) │
└───────────────┴───────────────┴───────────────┘
        │
        ▼
PostgreSQL (Results Storage via JDBC)
        │
        ▼
Tableau Visualization
├── XGBoost Line Chart
├── XGBoost Polar Chart
├── XGBoost Bar Chart
├── Demand vs Production Line Chart
├── Predicted Demand Stacked Area Chart
├── Actual vs Predicted Demand Line Chart
├── Production Trends Stacked Bar Chart
└── High/Low Production Frequency Chart

## 🛠️ Tech Stack

| Layer | Technology |![Uploading image.png…]()

|-------|------------|
| Data Source | USDA FeedGrains Dataset (Azure Data Lake Storage Gen2) |
| Processing | Apache Spark 3.5.5 (PySpark) |
| ML — Regression | XGBoost, PySpark Random Forest Regressor |
| ML — Classification | PySpark Random Forest Classifier |
| Storage | PostgreSQL 15 (via JDBC) |
| Language | Python 3.x |
| Notebook | Jupyter / IPython |
| Runner | Windows Batch (feedgrain.bat) |

---

## ⚙️ Prerequisites

### Software
- Python 3.8+
- Apache Spark 3.5.5 with Hadoop 3
- Java 11+ (required by Spark)
- PostgreSQL 15 (running on port 5433)
- PostgreSQL JDBC Driver (`postgresql-42.6.0.jar`)

### Python Packages
```bash
pip install pyspark xgboost scikit-learn pandas numpy matplotlib seaborn sqlalchemy psycopg2-binary
```

---

## 🔧 Configuration

Before running, update these values in `Feedgrain.py`:

### Azure Storage (Cell 1)
```python
storage_account_name = "your_storage_account"
storage_account_key  = "your_storage_key"
container_name       = "feedgrain"
```

> ⚠️ **Security Warning:** Never commit real storage keys to Git. Use environment variables or Azure Key Vault instead.

### PostgreSQL (Cells 9, 23, 37, 46)
```python
db_host     = "localhost"
db_port     = "5433"
db_name     = "postgres"
db_user     = "postgres"
db_password = "your_password"
```

### Spark JDBC Driver Path (Cells 9, 46)
```python
.config("spark.jars", "path/to/postgresql-42.6.0.jar")
```

---

## 🚀 How to Run

### Option 1 — Windows Batch File (Recommended)
```bat
double-click feedgrain.bat
```
or from command prompt:
```cmd
feedgrain.bat
```

### Option 2 — Python Directly
```bash
python Feedgrain.py
```

### Option 3 — Jupyter Notebook
```bash
jupyter notebook Code.ipynb
```

---

## 🗄️ PostgreSQL Output Tables

| Table | Description | Columns |
|-------|-------------|---------|
| `Feedgrain` | Cleaned raw data | All original columns |
| `xgboost` | Monthly price forecasts for 2026 | Year_ID, Month_ID, SC_Commodity_Desc, Predicted_Price |
| `demand` | Production vs Demand analysis | Year_ID, SC_Commodity_Desc, Production, Demand, Predicted_Demand |
| `Trends` | Production level classification | Year, Grain, Region, Avg_Production, High_Count, Low_Count |

---

## 📊 Dataset

- **Source:** [USDA ERS Feed Grains Database](https://www.ers.usda.gov/data-products/feed-grains-database/)
- **Grains covered:** Corn, Oats, Barley, Sorghum
- **Geography:** United States (and states)
- **Years:** ~1926 to present
- **Rows:** ~523,000

---

## 🔒 Security Note

This repository uses `.gitignore` to exclude sensitive files. **Do not commit:**
- Azure storage keys
- Database passwords
- `.env` files

Use environment variables for secrets:
```python
import os
storage_account_key = os.environ["AZURE_STORAGE_KEY"]
db_password         = os.environ["POSTGRES_PASSWORD"]
```

---

## 📌 .gitignore (Recommended)

Create a `.gitignore` file in your repository root:
```
*.pyc
__pycache__/
.env
*.log
*.jar
spark-warehouse/
derby.log
metastore_db/
```

---
## 📊 Results & Visualizations

### 1. XGBoost – Price Prediction

#### Model Performance Metrics

| Metric | Value |
|---|---|
| RMSE (Root Mean Squared Error) | 0.81 |
| MSE (Mean Squared Error) | 0.66 |
| MAPE (Mean Absolute Percentage Error) | 14.55% |
| R² (R-Squared) | 0.81 |

The model explains **81% of variance** in grain prices with an average prediction error of **14.55%**.

#### Predicted Prices for 2026 (Sample — PostgreSQL Output)

| Year | Month | Grain | Predicted Price |
|---|---|---|---|
| 2026 | January | Oats | ~$3.41 |
| 2026 | January | Corn | ~$4.83 |
| 2026 | January | Barley | ~$6.38 |
| 2026 | January | Sorghum | ~$3.75 |

> **Fig. 7** in the paper shows the full PostgreSQL output table with `Year_ID`, `Month_ID`, `SC_Commodity_Desc`, `commodity_index`, `sin_month`, `cos_month`, and `Predicted_Price` columns.

#### 📈 Graph 1 — Line Chart: Predicted Price Trends Over Time (Fig. 8)
- **What it shows:** Monthly predicted price trends for all four grains across 2026
- **Key findings:**
  - Barley → average ~$6.57 (highest)
  - Corn → stable ~$4.44
  - Sorghum → highest peak at ~$7.38
  - Oats → lowest at ~$3.41
  - <img width="1446" height="775" alt="7" src="https://github.com/user-attachments/assets/0e81ef9d-8ce8-407c-bf40-eec39d8ec032" />


#### 🌐 Graph 2 — Polar Chart: Monthly Seasonality in Predicted Commodity Prices (Fig. 9)
- **What it shows:** Seasonal price patterns of grains across months in a radial format
- **Key findings:**
  - Barley → consistently high throughout the year
  - Oats → near center, minimal price fluctuation
  - Sorghum → widest spread, especially mid-year, indicating strong seasonality
  - <img width="1431" height="771" alt="image" src="https://github.com/user-attachments/assets/85fe0212-1b5e-4402-ac7a-b1b26d935b8d" />


#### 📊 Graph 3 — Bar Chart: Monthly Average Price by Commodity (Fig. 10)
- **What it shows:** Average predicted prices by grain for each month of 2026
- **Key findings:**
  - Barley → highest average price across all months
  - Oats → low price with minimal fluctuation
  - Sorghum → most variation in July, August, October, and December
  - <img width="1437" height="765" alt="image" src="https://github.com/user-attachments/assets/7388e310-17a0-4f69-86e0-98de18c64477" />


---

### 2. Random Forest Regression – Demand Prediction

#### Model Performance

| Metric | Value |
|---|---|
| R² (R-Squared) | **0.96** |

The model captures **96% of demand variance**, showing very high prediction accuracy.

> **Fig. 10** in the paper shows the PostgreSQL output table with columns: `Year_ID`, `SC_Commodity_Desc`, `Production`, `Demand`, and `Predicted_Demand`.

#### 📈 Graph 4 — Line Chart: Production vs Demand of Major Feed Grains Over Time (Fig. 11)
- **What it shows:** Historical comparison of demand and production for all grains
- **Key findings:**
  - Corn dominates both demand and production, reaching **over 4 billion units** in recent years
  - Barley and Sorghum follow distantly
  - Oats show consistently low demand
  - <img width="1447" height="767" alt="image" src="https://github.com/user-attachments/assets/695c81c3-7600-49fe-b8eb-ff21e98bb7fd" />


#### 📊 Graph 5 — Stacked Area Chart: Predicted Demand Over Time (Fig. 12)
- **What it shows:** Predicted demand trajectory per grain from historical years onward
- **Key findings:**
  - Corn's predicted demand surges **beyond 4.5 billion units** after 1990
  - Other grains (Oats, Sorghum) remain relatively low
  - <img width="1450" height="767" alt="image" src="https://github.com/user-attachments/assets/03a4e279-a313-48d2-a4e5-e7ead66d804b" />


#### 📈 Graph 6 — Line Chart: Actual Demand vs Predicted Demand vs Production (Fig. 13)
- **What it shows:** Three-way comparison — production (purple), actual demand (green), predicted demand (blue)
- **Key findings:**
  - Actual demand and predicted demand lines are very closely aligned, validating model accuracy
  - Production remains significantly lower than demand
<img width="1447" height="775" alt="image" src="https://github.com/user-attachments/assets/ff976e67-da10-4f35-b1fe-0b4fb8211381" />

---

### 3. Random Forest Classifier – Production Trends

#### Model Performance

| Metric | Value |
|---|---|
| Model Accuracy | **91.95%** |
| F1 Score | **92.04%** |

The classifier achieves over **91% accuracy** and an F1 score of **92%**, indicating strong performance across all three production categories (Low, Medium, High).

> **Fig. 14** in the paper shows the PostgreSQL output table with `Year`, `Grain`, `Region`, `Avg_Production`, `High_Count`, and `Low_Count`.

#### 📊 Graph 7 — Stacked Bar Chart: Year-over-Year Grain Production (Fig. 15)
- **What it shows:** Contribution of each grain to total production per year from the mid-1800s onward
- **Key findings:**
  - Corn dominates production from the mid-1900s onwards and grows steadily
  - Oats and Barley become progressively smaller portions of total production
  - <img width="1447" height="772" alt="image" src="https://github.com/user-attachments/assets/e1e3705b-0452-4e45-a257-84451607bc14" />

#### 📈 Graph 8 — Line Chart: Average Grain Production Trends by Year (Fig. 15 — upper)
- **What it shows:** How average production per grain changed over time
- **Key findings:**
  - Corn shows steep growth since the 1950s
  - Barley, Oats, and Sorghum show a **declining trend**
  -  <img width="1443" height="780" alt="image" src="https://github.com/user-attachments/assets/a41cd394-2a71-43eb-ba0e-6fa04e6bb795" />


#### 📉 Graph 9 — Line Chart: Production vs High/Low Classification Frequency (Fig. 16)
- **What it shows:** Relationship between average production level and how often each grain was classified as High or Low
- **Key findings:**
  - Corn → consistently classified as High, never in Low category
  - Barley and Oats → many Low production records
  - <img width="1448" height="777" alt="image" src="https://github.com/user-attachments/assets/c5116a7e-2a01-4ee6-bbc0-4521f39967c4" />


---

## 📊 Tableau Dashboards

All model outputs are stored in **PostgreSQL** and connected to **Tableau** via a live database connection. The following dashboards were created:

| Dashboard | Data Source | Chart Types |
|---|---|---|
| **XGBoost Price Analysis** | `predicted_prices` table | Line chart, Polar chart, Bar chart |
| **Demand Prediction Analysis** | `demand_prediction` table | Line chart, Stacked area chart, Multi-line comparison |
| **Production Trends Analysis** | `production_trends` table | Stacked bar chart, Line chart, Scatter (High/Low frequency) |

### Connecting Tableau to PostgreSQL

1. Open Tableau Desktop
2. Select **Connect → PostgreSQL**
3. Enter server host, port (`5432`), database name, username, and password
4. Select the target table (`predicted_prices`, `demand_prediction`, or `production_trends`)
5. Drag fields into the worksheet to build visualizations

---

## ⚙️ Automation Script

The pipeline is automated using a **Windows Batch Script** (`Feedgrain.bat`) that triggers the main Python analysis script (`feedgrain.py`).

```bat
@echo off
echo Starting Feedgrain Analysis...
python feedgrain.py
echo Analysis complete.
```

**How to run:**
```bash
# Double-click Feedgrain.bat
# OR run from command prompt:
Feedgrain.bat
```

> **Figures 4, 5, and 6** in the paper show the automation flow and console output during execution, including the Spark session startup, schema display, model training, and PostgreSQL storage confirmation.

---


## 👤 Author

**Dnyanesh**  
Azure | PySpark | Machine Learning | PostgreSQL
