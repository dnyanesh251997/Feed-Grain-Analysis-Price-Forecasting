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

## 🛠️ Tech Stack

| Layer | Technology |
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

## 👤 Author

**Dnyanesh**  
Azure | PySpark | Machine Learning | PostgreSQL
