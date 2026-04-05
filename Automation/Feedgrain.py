#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, when, avg, count
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import when, col, sum as spark_sum
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, expr
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import trim
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, when
# Azure storage detail
storage_account_name = "Account_name"
storage_account_key = "Your_Key"
container_name = "feedgrain"

# Connecting spark with necessary configuration
spark = SparkSession.builder \
    .appName("AzureCSV") \
    .config(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key) \
    .getOrCreate()

# file path of Azure storage
read_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/Feedgrain/FeedGrains.csv"

# Read the file from azure storage
df = spark.read.csv(
    read_path,

    header=True,
    inferSchema=True
)

df.show()


# In[2]:


df.printSchema()


# In[3]:


for c in df.columns:
    if df.schema[c].dataType.simpleString() == 'string':
        df = df.withColumn(c, trim(col(c)))


# In[4]:


df = df.dropDuplicates()


# In[5]:


df = df.dropna(how='any')


# In[6]:


df = df.withColumn("Amount", col("Amount").cast(DoubleType()))
df = df.withColumn("Year_ID", col("Year_ID").cast(IntegerType()))


# In[7]:


# To make Amount column is cast to double
df = df.withColumn("Amount", col("Amount").cast("double"))

# Calculate Q1 and Q3
quantiles = df.approxQuantile("Amount", [0.25, 0.75], 0.01)
q1, q3 = quantiles[0], quantiles[1]
iqr = q3 - q1

# Define bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Filter outliers
outliers = df.filter((col("Amount") < lower_bound) |
                     (col("Amount") > upper_bound))


# In[8]:


# rows count after preprocessing
final_count = df.count()
print(f"Rows After Cleaning: {final_count}")


# In[9]:


spark = SparkSession.builder \
    .appName("WriteLargeDataToPostgres") \
    .config("spark.jars", "C:/Users/dnyan/SPARK/spark-3.5.5-bin-hadoop3/spark-3.5.5-bin-hadoop3/jars/postgresql-42.6.0.jar") \
    .getOrCreate()

# to save the data in postgresql using jdbc
df.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5433/postgres") \
    .option("dbtable", "Feedgrain") \
    .option("user", "postgres") \
    .option("password", "8192") \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()


# In[10]:


# How can historical trends  be leveraged to accurately forecast future feed grain prices using machine learning models?


# In[11]:


# In[12]:


targetgrains = ["Oats", "Corn", "Barley", "Sorghum"]
filtered = df.filter(
    (df["SC_Commodity_Desc"].isin(targetgrains)) &
    (df["SC_Frequency_Desc"] == "Annual") &
    (df["SC_GeographyIndented_Desc"] == "United States") &
    (df["SC_Attribute_Desc"].contains("Price"))
).select("Year_ID", "SC_Commodity_Desc", "Amount").dropna()


# In[13]:


# convert filter data to pandas because it does not xgboost in pyspark
df = filtered.toPandas()
# without xgboost4j-spark_2.12:1.6.1 which is not compatable with my spark version. so i use allternative to convert it into pandas frames


# In[14]:


dfrows = []

for _, row in df.iterrows():
    for month in range(1, 13):
        dfrows.append({
            "Year_ID": row["Year_ID"],
            "Month_ID": month,
            "SC_Commodity_Desc": row["SC_Commodity_Desc"],
            "Amount": row["Amount"]
        })

expanded_df = pd.DataFrame(dfrows)


# In[15]:


expanded_df["commodity_index"] = expanded_df["SC_Commodity_Desc"].astype(
    "category").cat.codes

# this add seasonal features for month
expanded_df["sin_month"] = np.sin(
    2 * np.pi * expanded_df["Month_ID"] / 12).round(2)
expanded_df["cos_month"] = np.cos(
    2 * np.pi * expanded_df["Month_ID"] / 12).round(2)


# In[16]:


# Here trainig the xgboost model
X = expanded_df[["Year_ID", "commodity_index", "sin_month", "cos_month"]]
y = expanded_df["Amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=100, objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)


# In[17]:


y_pred = model.predict(X_test)


# In[18]:


# Evalution metrics for xgboost model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

# printing the Output for metrics

print(f" RMSE : {rmse:.2f}")
print(f" MSE  : {mse:.2f}")
print(f" MAPE : {mape:.2f}%")
print(f" R²    : {r2:.2f}")


# In[19]:


startyear = datetime.now().year + 1
endyear = startyear  # future price for next Only one year


# In[20]:


future_dates = pd.date_range(
    start=f"{startyear}-01-01", end=f"{startyear}-12-01", freq="MS")

forecastdata = []
for grain in targetgrains:
    grain_index = expanded_df[expanded_df["SC_Commodity_Desc"]
                              == grain]["commodity_index"].iloc[0]
    for date in future_dates:
        forecastdata.append({
            "Year_ID": date.year,
            "Month_ID": date.month,
            "SC_Commodity_Desc": grain,
            "commodity_index": grain_index,
            "sin_month": np.round(np.sin(2 * np.pi * date.month / 12), 2),
            "cos_month": np.round(np.cos(2 * np.pi * date.month / 12), 2)
        })

future_df = pd.DataFrame(forecastdata)


# In[21]:


X_future = future_df[["Year_ID", "commodity_index", "sin_month", "cos_month"]]
future_df["Predicted_Price"] = model.predict(X_future)


# In[22]:


print(future_df.head(5))


# In[23]:


db_user = 'postgres'
db_password = '8192'
db_host = 'localhost'
db_port = '5433'
db_name = 'postgres'

engine = create_engine(
    f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
future_df.to_sql("xgboost", engine, if_exists="replace", index=False)
print("Forecasts saved to PostgreSQL!")


# In[24]:


# 2 How has the market demand for major feed grains evolved in relation to their production over the years?


# In[25]:


# In[26]:


df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5433/postgres") \
    .option("dbtable", "Feedgrain") \
    .option("user", "postgres") \
    .option("password", "8192") \
    .option("driver", "org.postgresql.Driver") \
    .load()


# In[27]:


feed_grains = ["Corn", "Barley", "Oats", "Sorghum"]
filtered_df = df.filter(col("SC_Commodity_Desc").isin(feed_grains))


# In[28]:


df_cat = filtered_df.withColumn(
    "Category",
    when(col("SC_Attribute_Desc").contains("Production"), "Production")
    .when(col("SC_Attribute_Desc").rlike("Use|Utilization|Export|Feed|Residual|Food|Seed|Industrial"), "Demand")
)


# In[29]:


# Filter relevant records
df_classified = df_cat.filter(col("Category").isNotNull())


# In[30]:


df_grouped = df_classified.groupBy("Year_ID", "SC_Commodity_Desc", "Category") \
    .agg(spark_sum("Amount").alias("Total")) \
    .groupBy("Year_ID", "SC_Commodity_Desc") \
    .pivot("Category").sum("Total") \
    .dropna()


# In[31]:


indexer = StringIndexer(inputCol="SC_Commodity_Desc", outputCol="GrainIndex")
df_indexed = indexer.fit(df_grouped).transform(df_grouped)


# In[32]:


assembler = VectorAssembler(
    inputCols=["Year_ID", "Production", "GrainIndex"], outputCol="features")
assembled = assembler.transform(df_indexed)


# In[33]:


train, test = assembled.randomSplit([0.8, 0.2], seed=42)

# 10. Train the model
rf = RandomForestRegressor(featuresCol="features",
                           labelCol="Demand", numTrees=100)
model = rf.fit(train)


# In[34]:


predictions = model.transform(test)
evaluator = RegressionEvaluator(
    labelCol="Demand", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"R² for combined model: {r2:.2f}")


# In[35]:


Demand = predictions.select("Year_ID", "SC_Commodity_Desc", "Production", "Demand", "prediction") \
    .withColumnRenamed("prediction", "Predicted_Demand")


# In[ ]:


print(Demand.head(5))


# In[37]:


Demand.write \
    .format("jdbc") \
    .option("url", f"jdbc:postgresql://{db_host}:{db_port}/{db_name}") \
    .option("dbtable", "demand") \
    .option("user", db_user) \
    .option("password", db_password) \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()

print("Forecasts saved to PostgreSQL!")


# In[38]:


# What are the patterns and trends in feed grain production over different seasons or years?


# In[39]:


# In[40]:


# Fix for Windows Python path
os.environ["PYSPARK_PYTHON"] = "python"


#  Filter for only Production
df = df.filter(
    (col("SC_Attribute_Desc") == "Production") &
    (col("SC_GeographyIndented_Desc") != "World less U.S.") &
    (col("SC_GroupCommod_Desc").isin("Oats", "Corn", "Sorghum", "Barley"))
)

#  Drop rows with missing key values
df = df.dropna(subset=[
    "SC_GroupCommod_Desc", "SC_GeographyIndented_Desc",
    "Year_ID", "Timeperiod_Desc", "Amount"
])


# In[41]:


# Compute quantile thresholds for categorizing production
quantiles = df.approxQuantile("Amount", [0.33, 0.66], 0.01)
low_thresh, high_thresh = quantiles

#  Add Production_Level column
df = df.withColumn(
    "Production_Level",
    when(col("Amount") <= low_thresh, "Low")
    .when(col("Amount") <= high_thresh, "Medium")
    .otherwise("High")
)


# In[42]:


# Feature preparation
indexers = [
    StringIndexer(inputCol="SC_GroupCommod_Desc", outputCol="GrainIndex"),
    StringIndexer(inputCol="SC_GeographyIndented_Desc",
                  outputCol="RegionIndex"),
    StringIndexer(inputCol="Timeperiod_Desc", outputCol="SeasonIndex"),
    StringIndexer(inputCol="Production_Level", outputCol="label")
]

assembler = VectorAssembler(
    inputCols=["GrainIndex", "RegionIndex", "SeasonIndex", "Year_ID"],
    outputCol="features"
)


# In[43]:


#  Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features",
                            labelCol="label", numTrees=100)

# Step 9: Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Step 10: Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 11: Fit model
model = pipeline.fit(train_data)

# Step 12: Predict
predictions = model.transform(test_data)


# In[44]:


evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)


f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

# Print metrics
print(f"\n Model Accuracy: {accuracy:.2%}")
print(f" F1 Score: {f1_score:.2%}")

#  Grouped analysis by Year, Grain, Region
agg_df = df.groupBy("Year_ID", "SC_GroupCommod_Desc", "SC_GeographyIndented_Desc").agg(
    avg("Amount").alias("Avg_Production"),
    count(when(col("Production_Level") == "High", True)).alias("High_Count"),
    count(when(col("Production_Level") == "Low", True)).alias("Low_Count")
)


# In[45]:


# Rename columns for output
agg_df = agg_df.withColumnRenamed("Year_ID", "Year") \
               .withColumnRenamed("SC_GroupCommod_Desc", "Grain") \
               .withColumnRenamed("SC_GeographyIndented_Desc", "Region")

# Optional: Show aggregated results
agg_df.show()


# In[46]:


# Create Spark Session with PostgreSQL JDBC driver
spark = SparkSession.builder \
    .appName("ReadDataFromPostgres") \
    .config("spark.jars", "C:/Users/dnyan/SPARK/spark-3.5.5-bin-hadoop3/spark-3.5.5-bin-hadoop3/jars/postgresql-42.6.0.jar") \
    .getOrCreate()
agg_df.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5433/postgres") \
    .option("dbtable", "Trends") \
    .option("user", "postgres") \
    .option("password", "8192") \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()
