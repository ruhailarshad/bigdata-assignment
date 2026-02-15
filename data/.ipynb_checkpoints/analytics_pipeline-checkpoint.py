"""
RetailChain Big Data Analytics Pipeline
Complete PySpark script for data ingestion, cleaning, analytics, and ML
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import os

def create_spark_session():
    """Initialize Spark session with HDFS configuration"""
    spark = SparkSession.builder \
        .appName("RetailChain-DataCo-POC") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

def load_and_clean_data(spark, hdfs_path):
    """Load CSV from HDFS and perform data cleaning"""
    print("=" * 60)
    print("STEP 1: Loading data from HDFS...")
    print("=" * 60)
    
    df = spark.read.csv(
        hdfs_path, 
        header=True, 
        inferSchema=True, 
        encoding="ISO-8859-1"
    )
    
    print(f"Initial row count: {df.count()}")
    print(f"Initial column count: {len(df.columns)}")
    
    print("\nSTEP 2: Data cleaning and transformation...")
    print("-" * 60)
    
    # Parse dates
    df2 = df.withColumn(
        "order_dt", 
        F.to_timestamp(F.col("order date (DateOrders)"), "M/d/yyyy H:mm")
    ).withColumn(
        "ship_dt", 
        F.to_timestamp(F.col("shipping date (DateOrders)"), "M/d/yyyy H:mm")
    )
    
    # Cast numeric columns
    num_cols = [
        "Sales", 
        "Order Profit Per Order", 
        "Late_delivery_risk",
        "Days for shipping (real)", 
        "Days for shipment (scheduled)"
    ]
    
    for c in num_cols:
        if c in df2.columns:
            df2 = df2.withColumn(c, F.col(c).cast("double"))
    
    print("✓ Date parsing completed")
    print("✓ Numeric type casting completed")
    
    # Show sample data
    print("\nSample data (first 5 rows):")
    df2.select("order_dt", "Sales", "Order Profit Per Order", "Late_delivery_risk").show(5, truncate=False)
    
    print("\nData schema:")
    df2.printSchema()
    
    return df2

def write_curated_data(df, curated_path):
    """Write cleaned data to Parquet format"""
    print("\n" + "=" * 60)
    print("STEP 3: Writing curated data to Parquet...")
    print("=" * 60)
    
    df.write.mode("overwrite").parquet(curated_path)
    print(f"✓ Curated data written to: {curated_path}")
    
    return df

def monthly_trend_analysis(df):
    """Analyze monthly sales and profit trends"""
    print("\n" + "=" * 60)
    print("ANALYTICS 1: Monthly Sales/Profit Trend")
    print("=" * 60)
    
    monthly = (df
        .withColumn("order_month", F.date_format("order_dt", "yyyy-MM"))
        .groupBy("order_month")
        .agg(
            F.countDistinct("Order Id").alias("orders"),
            F.sum("Sales").alias("sales"),
            F.sum("Order Profit Per Order").alias("profit"),
            F.avg("Late_delivery_risk").alias("late_rate")
        )
        .orderBy("order_month")
    )
    
    print("\nMonthly KPIs:")
    monthly.show(12, truncate=False)
    
    # Convert to Pandas for visualization
    monthly_pd = monthly.toPandas()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(monthly_pd["order_month"], monthly_pd["sales"], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.title("Monthly Sales Trend")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(monthly_pd["order_month"], monthly_pd["profit"], marker='o', color='green')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Profit")
    plt.title("Monthly Profit Trend")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/work/outputs/monthly_trends.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved: outputs/monthly_trends.png")
    
    return monthly

def category_analysis(df):
    """Analyze category contribution to sales and profit"""
    print("\n" + "=" * 60)
    print("ANALYTICS 2: Category Contribution Analysis")
    print("=" * 60)
    
    cat_kpis = (df.groupBy("Category Name")
        .agg(
            F.sum("Sales").alias("sales"),
            F.sum("Order Profit Per Order").alias("profit"),
            F.countDistinct("Order Id").alias("orders")
        )
        .orderBy(F.desc("sales"))
    )
    
    print("\nTop 10 Categories by Sales:")
    cat_kpis.show(10, truncate=False)
    
    # Visualization
    cat_pd = cat_kpis.limit(10).toPandas()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(range(len(cat_pd)), cat_pd["sales"], color='steelblue')
    plt.yticks(range(len(cat_pd)), cat_pd["Category Name"])
    plt.xlabel("Total Sales")
    plt.title("Top 10 Categories by Sales")
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(cat_pd)), cat_pd["profit"], color='darkgreen')
    plt.yticks(range(len(cat_pd)), cat_pd["Category Name"])
    plt.xlabel("Total Profit")
    plt.title("Top 10 Categories by Profit")
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/work/outputs/category_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved: outputs/category_analysis.png")
    
    return cat_kpis

def shipping_mode_analysis(df):
    """Analyze delivery performance by shipping mode"""
    print("\n" + "=" * 60)
    print("ANALYTICS 3: Shipping Mode Performance Analysis")
    print("=" * 60)
    
    ship_mode = (df.groupBy("Shipping Mode")
        .agg(
            F.count("*").alias("total_orders"),
            F.avg("Late_delivery_risk").alias("late_rate"),
            F.avg("Days for shipping (real)").alias("avg_real_ship"),
            F.avg("Days for shipment (scheduled)").alias("avg_sched")
        )
        .orderBy(F.desc("late_rate"))
    )
    
    print("\nShipping Mode Performance:")
    ship_mode.show(truncate=False)
    
    # Visualization
    ship_pd = ship_mode.toPandas()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(ship_pd["Shipping Mode"], ship_pd["late_rate"], color='coral')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Late Delivery Rate")
    plt.title("Late Delivery Risk by Shipping Mode")
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    x = range(len(ship_pd))
    width = 0.35
    plt.bar([i - width/2 for i in x], ship_pd["avg_real_ship"], width, label='Actual', color='steelblue')
    plt.bar([i + width/2 for i in x], ship_pd["avg_sched"], width, label='Scheduled', color='orange')
    plt.xticks(x, ship_pd["Shipping Mode"], rotation=45, ha="right")
    plt.ylabel("Days")
    plt.title("Actual vs Scheduled Shipping Days")
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/work/outputs/shipping_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved: outputs/shipping_analysis.png")
    
    return ship_mode

def ml_late_delivery_prediction(df):
    """Build ML model to predict late delivery risk"""
    print("\n" + "=" * 60)
    print("MACHINE LEARNING: Late Delivery Risk Prediction")
    print("=" * 60)
    
    # Prepare ML dataset
    ml_df = df.select(
        col("Late_delivery_risk").cast("int").alias("label"),
        col("Days for shipping (real)").alias("ship_real"),
        col("Days for shipment (scheduled)").alias("ship_sched"),
        col("Shipping Mode").alias("ship_mode"),
        col("Category Name").alias("category"),
        col("Order Region").alias("region"),
        col("Sales").alias("sales")
    ).na.drop()
    
    print(f"ML dataset size: {ml_df.count()} rows")
    
    # Feature engineering
    cat_cols = ["ship_mode", "category", "region"]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") 
        for c in cat_cols
    ]
    
    enc = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_oh" for c in cat_cols]
    )
    
    assembler = VectorAssembler(
        inputCols=["ship_real", "ship_sched", "sales"] + [f"{c}_oh" for c in cat_cols],
        outputCol="features"
    )
    
    # Build pipeline
    prep_pipeline = Pipeline(stages=indexers + [enc, assembler])
    prep_model = prep_pipeline.fit(ml_df)
    ready_df = prep_model.transform(ml_df).select("label", "features")
    
    # Split data
    train, test = ready_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set: {train.count()} rows")
    print(f"Test set: {test.count()} rows")
    
    # Train model
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(labelCol="label", featuresCol="features")
    model = lr_model.fit(train)
    
    # Evaluate
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)
    
    print(f"\n✓ Model trained successfully!")
    print(f"✓ AUC Score: {auc:.4f}")
    
    # Show sample predictions
    print("\nSample predictions:")
    predictions.select("label", "prediction", "probability").show(10, truncate=False)
    
    return model, auc

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("RETAILCHAIN BIG DATA ANALYTICS PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('/home/jovyan/work/outputs', exist_ok=True)
    
    # Initialize Spark
    spark = create_spark_session()
    
    # HDFS paths
    hdfs_raw_path = "hdfs://namenode:8020/data/retailchain/raw/DataCoSupplyChainDataset.csv"
    hdfs_curated_path = "hdfs://namenode:8020/data/retailchain/curated/dataco_parquet"
    
    try:
        # Load and clean data
        df_cleaned = load_and_clean_data(spark, hdfs_raw_path)
        
        # Write curated data
        write_curated_data(df_cleaned, hdfs_curated_path)
        
        # Run analytics
        monthly_trend_analysis(df_cleaned)
        category_analysis(df_cleaned)
        shipping_mode_analysis(df_cleaned)
        
        # Optional ML
        try:
            ml_model, auc_score = ml_late_delivery_prediction(df_cleaned)
            print(f"\n{'='*60}")
            print("SUMMARY: ML Model AUC Score = {:.4f}".format(auc_score))
            print("="*60)
        except Exception as e:
            print(f"\nML pipeline skipped due to error: {str(e)}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files saved in: outputs/")
        print("- monthly_trends.png")
        print("- category_analysis.png")
        print("- shipping_analysis.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
