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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Try to import seaborn for enhanced visualizations (optional)
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # seaborn not available, will use matplotlib defaults

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

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

def regional_analysis(df):
    """Analyze sales and profit by region"""
    print("\n" + "=" * 60)
    print("ANALYTICS 4: Regional Performance Analysis")
    print("=" * 60)
    
    regional = (df.groupBy("Order Region")
        .agg(
            F.sum("Sales").alias("total_sales"),
            F.sum("Order Profit Per Order").alias("total_profit"),
            F.countDistinct("Order Id").alias("total_orders"),
            F.avg("Order Profit Per Order").alias("avg_profit_per_order"),
            F.avg("Late_delivery_risk").alias("late_delivery_rate")
        )
        .withColumn("profit_margin", (F.col("total_profit") / F.col("total_sales")) * 100)
        .orderBy(F.desc("total_sales"))
    )
    
    print("\nRegional Performance Summary:")
    regional.show(truncate=False)
    
    # Convert to Pandas for visualization
    reg_pd = regional.toPandas()
    
    # Create comprehensive regional visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sales by Region (Bar Chart)
    axes[0, 0].barh(range(len(reg_pd)), reg_pd["total_sales"], color='steelblue')
    axes[0, 0].set_yticks(range(len(reg_pd)))
    axes[0, 0].set_yticklabels(reg_pd["Order Region"])
    axes[0, 0].set_xlabel("Total Sales ($)")
    axes[0, 0].set_title("Total Sales by Region")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Profit Margin by Region
    axes[0, 1].barh(range(len(reg_pd)), reg_pd["profit_margin"], color='darkgreen')
    axes[0, 1].set_yticks(range(len(reg_pd)))
    axes[0, 1].set_yticklabels(reg_pd["Order Region"])
    axes[0, 1].set_xlabel("Profit Margin (%)")
    axes[0, 1].set_title("Profit Margin by Region")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Orders vs Late Delivery Rate (Scatter)
    axes[1, 0].scatter(reg_pd["total_orders"], reg_pd["late_delivery_rate"], 
                       s=reg_pd["total_sales"]/10000, alpha=0.6, c=reg_pd["profit_margin"], 
                       cmap='RdYlGn', edgecolors='black')
    axes[1, 0].set_xlabel("Total Orders")
    axes[1, 0].set_ylabel("Late Delivery Rate")
    axes[1, 0].set_title("Orders vs Late Delivery (bubble size = sales)")
    axes[1, 0].grid(True, alpha=0.3)
    for i, region in enumerate(reg_pd["Order Region"]):
        axes[1, 0].annotate(region, (reg_pd["total_orders"].iloc[i], 
                                     reg_pd["late_delivery_rate"].iloc[i]),
                           fontsize=8, alpha=0.7)
    
    # 4. Sales Distribution (Pie Chart)
    axes[1, 1].pie(reg_pd["total_sales"], labels=reg_pd["Order Region"], 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title("Sales Distribution by Region")
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/work/outputs/regional_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved: outputs/regional_analysis.png")
    
    return regional

def customer_segment_analysis(df):
    """Analyze customer segment performance"""
    print("\n" + "=" * 60)
    print("ANALYTICS 5: Customer Segment Analysis")
    print("=" * 60)
    
    if "Customer Segment" in df.columns:
        segment = (df.groupBy("Customer Segment")
            .agg(
                F.countDistinct("Order Id").alias("total_orders"),
                F.sum("Sales").alias("total_sales"),
                F.sum("Order Profit Per Order").alias("total_profit"),
                F.avg("Sales").alias("avg_order_value"),
                F.avg("Order Profit Per Order").alias("avg_profit_per_order")
            )
            .orderBy(F.desc("total_sales"))
        )
        
        print("\nCustomer Segment Performance:")
        segment.show(truncate=False)
        
        seg_pd = segment.toPandas()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Sales by Segment
        axes[0].bar(range(len(seg_pd)), seg_pd["total_sales"], color='coral')
        axes[0].set_xticks(range(len(seg_pd)))
        axes[0].set_xticklabels(seg_pd["Customer Segment"], rotation=45, ha='right')
        axes[0].set_ylabel("Total Sales ($)")
        axes[0].set_title("Total Sales by Customer Segment")
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Average Order Value
        axes[1].bar(range(len(seg_pd)), seg_pd["avg_order_value"], color='steelblue')
        axes[1].set_xticks(range(len(seg_pd)))
        axes[1].set_xticklabels(seg_pd["Customer Segment"], rotation=45, ha='right')
        axes[1].set_ylabel("Average Order Value ($)")
        axes[1].set_title("Average Order Value by Segment")
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Orders Count
        axes[2].bar(range(len(seg_pd)), seg_pd["total_orders"], color='darkgreen')
        axes[2].set_xticks(range(len(seg_pd)))
        axes[2].set_xticklabels(seg_pd["Customer Segment"], rotation=45, ha='right')
        axes[2].set_ylabel("Total Orders")
        axes[2].set_title("Total Orders by Segment")
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/home/jovyan/work/outputs/customer_segment_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ Chart saved: outputs/customer_segment_analysis.png")
        
        return segment
    else:
        print("Customer Segment column not found, skipping analysis")
        return None

def time_analysis(df):
    """Analyze patterns by time (hour, day of week)"""
    print("\n" + "=" * 60)
    print("ANALYTICS 6: Time-Based Pattern Analysis")
    print("=" * 60)
    
    time_df = (df
        .withColumn("order_hour", F.hour("order_dt"))
        .withColumn("order_day", F.date_format("order_dt", "EEEE"))
        .withColumn("order_day_num", F.dayofweek("order_dt"))
    )
    
    # Hourly analysis
    hourly = (time_df.groupBy("order_hour")
        .agg(
            F.count("*").alias("order_count"),
            F.sum("Sales").alias("total_sales"),
            F.avg("Sales").alias("avg_sales")
        )
        .orderBy("order_hour")
    )
    
    print("\nHourly Order Patterns:")
    hourly.show(24, truncate=False)
    
    # Day of week analysis
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = (time_df.groupBy("order_day", "order_day_num")
        .agg(
            F.count("*").alias("order_count"),
            F.sum("Sales").alias("total_sales"),
            F.avg("Sales").alias("avg_sales")
        )
        .orderBy("order_day_num")
    )
    
    print("\nDay of Week Patterns:")
    daily.show(truncate=False)
    
    # Visualizations
    hourly_pd = hourly.toPandas()
    daily_pd = daily.toPandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Hourly order count
    axes[0, 0].plot(hourly_pd["order_hour"], hourly_pd["order_count"], 
                    marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("Order Count")
    axes[0, 0].set_title("Orders by Hour of Day")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Hourly sales
    axes[0, 1].bar(hourly_pd["order_hour"], hourly_pd["total_sales"], color='coral')
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Total Sales ($)")
    axes[0, 1].set_title("Sales by Hour of Day")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xticks(range(0, 24, 2))
    
    # Day of week order count
    axes[1, 0].bar(range(len(daily_pd)), daily_pd["order_count"], color='darkgreen')
    axes[1, 0].set_xticks(range(len(daily_pd)))
    axes[1, 0].set_xticklabels(daily_pd["order_day"], rotation=45, ha='right')
    axes[1, 0].set_ylabel("Order Count")
    axes[1, 0].set_title("Orders by Day of Week")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Day of week sales
    axes[1, 1].bar(range(len(daily_pd)), daily_pd["total_sales"], color='purple')
    axes[1, 1].set_xticks(range(len(daily_pd)))
    axes[1, 1].set_xticklabels(daily_pd["order_day"], rotation=45, ha='right')
    axes[1, 1].set_ylabel("Total Sales ($)")
    axes[1, 1].set_title("Sales by Day of Week")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/work/outputs/time_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved: outputs/time_analysis.png")
    
    return hourly, daily

def product_performance_analysis(df):
    """Analyze top and bottom performing products"""
    print("\n" + "=" * 60)
    print("ANALYTICS 7: Product Performance Analysis")
    print("=" * 60)
    
    if "Product Name" in df.columns:
        product = (df.groupBy("Product Name", "Category Name")
            .agg(
                F.count("*").alias("times_ordered"),
                F.sum("Sales").alias("total_sales"),
                F.sum("Order Profit Per Order").alias("total_profit"),
                F.avg("Sales").alias("avg_sale_price")
            )
            .withColumn("profit_margin", (F.col("total_profit") / F.col("total_sales")) * 100)
        )
        
        # Top 10 products by sales
        top_products = product.orderBy(F.desc("total_sales")).limit(10)
        print("\nTop 10 Products by Sales:")
        top_products.select("Product Name", "Category Name", "total_sales", 
                           "times_ordered", "profit_margin").show(truncate=30)
        
        # Top 10 products by profit
        top_profit = product.orderBy(F.desc("total_profit")).limit(10)
        print("\nTop 10 Products by Profit:")
        top_profit.select("Product Name", "Category Name", "total_profit", 
                         "times_ordered", "profit_margin").show(truncate=30)
        
        # Bottom 10 products (lowest sales)
        bottom_products = product.orderBy("total_sales").limit(10)
        print("\nBottom 10 Products by Sales:")
        bottom_products.select("Product Name", "Category Name", "total_sales", 
                              "times_ordered").show(truncate=30)
        
        # Visualization
        top_pd = top_products.toPandas()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Top products by sales
        axes[0].barh(range(len(top_pd)), top_pd["total_sales"], color='steelblue')
        axes[0].set_yticks(range(len(top_pd)))
        axes[0].set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                                 for name in top_pd["Product Name"]], fontsize=8)
        axes[0].set_xlabel("Total Sales ($)")
        axes[0].set_title("Top 10 Products by Sales")
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Top products by profit
        top_profit_pd = top_profit.toPandas()
        axes[1].barh(range(len(top_profit_pd)), top_profit_pd["total_profit"], color='darkgreen')
        axes[1].set_yticks(range(len(top_profit_pd)))
        axes[1].set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                                 for name in top_profit_pd["Product Name"]], fontsize=8)
        axes[1].set_xlabel("Total Profit ($)")
        axes[1].set_title("Top 10 Products by Profit")
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('/home/jovyan/work/outputs/product_performance.png', dpi=150, bbox_inches='tight')
        print("\n✓ Chart saved: outputs/product_performance.png")
        
        return product
    else:
        print("Product Name column not found, skipping analysis")
        return None

def order_status_analysis(df):
    """Analyze order status distribution"""
    print("\n" + "=" * 60)
    print("ANALYTICS 8: Order Status Analysis")
    print("=" * 60)
    
    if "Order Status" in df.columns:
        status = (df.groupBy("Order Status")
            .agg(
                F.count("*").alias("order_count"),
                F.sum("Sales").alias("total_sales"),
                F.sum("Order Profit Per Order").alias("total_profit"),
                F.avg("Late_delivery_risk").alias("avg_late_risk")
            )
            .orderBy(F.desc("order_count"))
        )
        
        print("\nOrder Status Distribution:")
        status.show(truncate=False)
        
        status_pd = status.toPandas()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Order count by status (Pie)
        axes[0].pie(status_pd["order_count"], labels=status_pd["Order Status"], 
                   autopct='%1.1f%%', startangle=90)
        axes[0].set_title("Order Count by Status")
        
        # Sales by status (Bar)
        axes[1].bar(range(len(status_pd)), status_pd["total_sales"], color='coral')
        axes[1].set_xticks(range(len(status_pd)))
        axes[1].set_xticklabels(status_pd["Order Status"], rotation=45, ha='right')
        axes[1].set_ylabel("Total Sales ($)")
        axes[1].set_title("Sales by Order Status")
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Late delivery risk by status
        axes[2].bar(range(len(status_pd)), status_pd["avg_late_risk"], color='red')
        axes[2].set_xticks(range(len(status_pd)))
        axes[2].set_xticklabels(status_pd["Order Status"], rotation=45, ha='right')
        axes[2].set_ylabel("Average Late Delivery Risk")
        axes[2].set_title("Late Delivery Risk by Status")
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/home/jovyan/work/outputs/order_status_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ Chart saved: outputs/order_status_analysis.png")
        
        return status
    else:
        print("Order Status column not found, skipping analysis")
        return None

def summary_statistics_table(df):
    """Generate comprehensive summary statistics table"""
    print("\n" + "=" * 60)
    print("ANALYTICS 9: Summary Statistics")
    print("=" * 60)
    
    summary = df.select(
        F.count("*").alias("total_records"),
        F.countDistinct("Order Id").alias("total_orders"),
        F.countDistinct("Customer Id").alias("total_customers"),
        F.sum("Sales").alias("total_sales"),
        F.avg("Sales").alias("avg_sale"),
        F.stddev("Sales").alias("stddev_sales"),
        F.sum("Order Profit Per Order").alias("total_profit"),
        F.avg("Order Profit Per Order").alias("avg_profit"),
        F.avg("Late_delivery_risk").alias("avg_late_delivery_rate"),
        F.avg("Days for shipping (real)").alias("avg_shipping_days")
    )
    
    print("\nOverall Summary Statistics:")
    summary.show(truncate=False)
    
    # Category-wise summary
    if "Category Name" in df.columns:
        cat_summary = (df.groupBy("Category Name")
            .agg(
                F.count("*").alias("records"),
                F.countDistinct("Order Id").alias("orders"),
                F.sum("Sales").alias("sales"),
                F.sum("Order Profit Per Order").alias("profit"),
                F.avg("Order Profit Per Order").alias("avg_profit_per_order")
            )
            .withColumn("profit_margin_pct", (F.col("profit") / F.col("sales")) * 100)
            .orderBy(F.desc("sales"))
        )
        
        print("\nCategory-wise Summary:")
        cat_summary.show(truncate=False)
        
        # Save summary table as CSV visualization
        cat_summary_pd = cat_summary.toPandas()
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(cat_summary_pd) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=cat_summary_pd.values,
                        colLabels=cat_summary_pd.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(cat_summary_pd.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title("Category-wise Summary Statistics", fontsize=14, fontweight='bold', pad=20)
        plt.savefig('/home/jovyan/work/outputs/summary_statistics.png', dpi=150, bbox_inches='tight')
        print("\n✓ Chart saved: outputs/summary_statistics.png")
        
        return summary, cat_summary
    
    return summary, None

def correlation_analysis(df):
    """Analyze correlations between numeric variables"""
    print("\n" + "=" * 60)
    print("ANALYTICS 10: Correlation Analysis")
    print("=" * 60)
    
    numeric_cols = ["Sales", "Order Profit Per Order", "Late_delivery_risk",
                    "Days for shipping (real)", "Days for shipment (scheduled)"]
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(available_cols) >= 2:
        corr_df = df.select(available_cols).toPandas()
        correlation_matrix = corr_df.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        # Heatmap visualization
        plt.figure(figsize=(10, 8))
        if HAS_SEABORN:
            sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        else:
            # Fallback to matplotlib imshow
            im = plt.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            # Add text annotations
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}', 
                            ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
        plt.title("Correlation Matrix: Key Metrics", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/home/jovyan/work/outputs/correlation_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ Chart saved: outputs/correlation_analysis.png")
        
        return correlation_matrix
    else:
        print("Not enough numeric columns for correlation analysis")
        return None

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
        regional_analysis(df_cleaned)
        customer_segment_analysis(df_cleaned)
        time_analysis(df_cleaned)
        product_performance_analysis(df_cleaned)
        order_status_analysis(df_cleaned)
        summary_statistics_table(df_cleaned)
        correlation_analysis(df_cleaned)
        
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
        print("- regional_analysis.png")
        print("- customer_segment_analysis.png")
        print("- time_analysis.png")
        print("- product_performance.png")
        print("- order_status_analysis.png")
        print("- summary_statistics.png")
        print("- correlation_analysis.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
