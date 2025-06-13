
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, desc, date_format, window, expr
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
import matplotlib.pyplot as plt
import seaborn as sns


# 创建Spark会话
def create_spark_session():
    spark = SparkSession.builder \
        .appName("EcommerceAnalytics") \
        .config("spark.sql.warehouse.dir", "hdfs:///user/hive/warehouse") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .enableHiveSupport() \
        .getOrCreate()

    print("Spark会话已创建")
    return spark


# 定义数据模式
def define_schema():
    # 订单数据模式
    order_schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("quantity", IntegerType(), False),
        StructField("price", FloatType(), False),
        StructField("total_amount", FloatType(), False),
        StructField("timestamp", TimestampType(), False),
        StructField("status", StringType(), False)
    ])

    # 用户数据模式
    user_schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("username", StringType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", TimestampType(), True),
        StructField("last_login", TimestampType(), True),
        StructField("user_level", StringType(), True)
    ])

    # 产品数据模式
    product_schema = StructType([
        StructField("product_id", StringType(), False),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", FloatType(), True),
        StructField("inventory", IntegerType(), True)
    ])

    return order_schema, user_schema, product_schema


# 加载数据
def load_data(spark, order_schema):
    # 从CSV加载订单数据
    orders_df = spark.read.csv("hdfs:///user/ecommerce/data/orders.csv",
                               header=True,
                               schema=order_schema)

    # 注册为临时视图
    orders_df.createOrReplaceTempView("orders")

    print("数据加载完成")
    print(f"订单数据记录数: {orders_df.count()}")

    # 显示数据示例
    print("订单数据示例:")
    orders_df.show(5)

    return orders_df


# 销售分析
def sales_analysis(spark, orders_df):
    print("\n执行销售分析...")

    # 每日销售额统计
    daily_sales = orders_df.filter(col("status") == "completed") \
        .withColumn("order_date", date_format(col("timestamp"), "yyyy-MM-dd")) \
        .groupBy("order_date") \
        .agg(sum("total_amount").alias("daily_sales"),
             count("order_id").alias("orders_count"))

    print("每日销售额统计:")
    daily_sales.orderBy(desc("order_date")).show(5)

    # 产品类别销售分析
    category_sales = spark.sql("""
        SELECT 
            p.category,
            SUM(o.total_amount) as category_sales,
            COUNT(DISTINCT o.order_id) as orders_count,
            COUNT(DISTINCT o.user_id) as customers_count
        FROM 
            orders o
        JOIN 
            products p ON o.product_id = p.product_id
        WHERE 
            o.status = 'completed'
        GROUP BY 
            p.category
        ORDER BY 
            category_sales DESC
    """)

    print("产品类别销售分析:")
    category_sales.show()

    # 用户消费等级分析
    user_spending = spark.sql("""
        SELECT 
            u.user_level,
            COUNT(DISTINCT u.user_id) as users_count,
            SUM(o.total_amount) as total_spending,
            AVG(o.total_amount) as avg_order_value
        FROM 
            orders o
        JOIN 
            users u ON o.user_id = u.user_id
        WHERE 
            o.status = 'completed'
        GROUP BY 
            u.user_level
        ORDER BY 
            avg_order_value DESC
    """)

    print("用户消费等级分析:")
    user_spending.show()

    return daily_sales, category_sales, user_spending


# RFM客户价值分析
def rfm_analysis(spark):
    print("\n执行RFM客户价值分析...")

    rfm_analysis = spark.sql("""
        WITH user_metrics AS (
            SELECT 
                user_id,
                MAX(timestamp) as last_purchase_date,
                COUNT(DISTINCT order_id) as frequency,
                SUM(total_amount) as monetary
            FROM 
                orders
            WHERE 
                status = 'completed'
            GROUP BY 
                user_id
        ),
        rfm_scores AS (
            SELECT 
                user_id,
                DATEDIFF(CURRENT_DATE, last_purchase_date) as recency_days,
                frequency,
                monetary,
                NTILE(5) OVER (ORDER BY DATEDIFF(CURRENT_DATE, last_purchase_date) DESC) as r_score,
                NTILE(5) OVER (ORDER BY frequency ASC) as f_score,
                NTILE(5) OVER (ORDER BY monetary ASC) as m_score
            FROM 
                user_metrics
        )
        SELECT 
            user_id,
            recency_days,
            frequency,
            monetary,
            r_score,
            f_score,
            m_score,
            CONCAT(r_score, f_score, m_score) as rfm_score,
            CASE 
                WHEN (r_score >= 4 AND f_score >= 4 AND m_score >= 4) THEN '高价值客户'
                WHEN (r_score >= 3 AND f_score >= 3 AND m_score >= 3) THEN '中高价值客户'
                WHEN (r_score >= 3 AND f_score >= 1 AND m_score >= 2) THEN '潜力客户'
                WHEN (r_score <= 2 AND f_score <= 2 AND m_score <= 2) THEN '流失风险客户'
                ELSE '一般价值客户'
            END as customer_segment
        FROM 
            rfm_scores
        ORDER BY 
            monetary DESC
    """)

    print("RFM客户价值分析结果:")
    rfm_analysis.show(10)

    # 客户细分统计
    customer_segments = rfm_analysis.groupBy("customer_segment") \
        .agg(count("user_id").alias("customers_count"),
             sum("monetary").alias("total_spending"),
             avg("monetary").alias("avg_spending"),
             avg("frequency").alias("avg_frequency"))

    print("客户细分统计:")
    customer_segments.orderBy(desc("total_spending")).show()

    return rfm_analysis, customer_segments


# 商品关联分析
def product_association(spark):
    print("\n执行商品关联分析...")

    # 使用Spark SQL进行关联规则挖掘
    basket_analysis = spark.sql("""
        WITH order_products AS (
            SELECT 
                o1.order_id,
                o1.product_id as product1,
                o2.product_id as product2,
                p1.product_name as product1_name,
                p2.product_name as product2_name
            FROM 
                orders o1
            JOIN 
                orders o2 ON o1.order_id = o2.order_id AND o1.product_id < o2.product_id
            JOIN 
                products p1 ON o1.product_id = p1.product_id
            JOIN 
                products p2 ON o2.product_id = p2.product_id
            WHERE 
                o1.status = 'completed' AND o2.status = 'completed'
        ),
        product_pairs AS (
            SELECT 
                product1,
                product2,
                product1_name,
                product2_name,
                COUNT(*) as pair_frequency
            FROM 
                order_products
            GROUP BY 
                product1, product2, product1_name, product2_name
        )
        SELECT 
            product1,
            product2,
            product1_name,
            product2_name,
            pair_frequency,
            pair_frequency / (
                SELECT COUNT(DISTINCT order_id) FROM orders WHERE status = 'completed'
            ) as support,
            pair_frequency / (
                SELECT COUNT(*) FROM orders WHERE product_id = product1 AND status = 'completed'
            ) as confidence
        FROM 
            product_pairs
        WHERE 
            pair_frequency >= 5
        ORDER BY 
            pair_frequency DESC
    """)

    print("商品关联分析结果:")
    basket_analysis.show(10)

    return basket_analysis


# 可视化分析结果
def visualize_results(daily_sales_df, category_sales_df):
    # 转换为Pandas DataFrame进行可视化
    daily_sales_pd = daily_sales_df.toPandas()
    category_sales_pd = category_sales_df.toPandas()

    # 设置图表风格
    plt.style.use('ggplot')

    # 每日销售额趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales_pd['order_date'], daily_sales_pd['daily_sales'], marker='o', linewidth=2)
    plt.title('每日销售额趋势', fontsize=15)
    plt.xlabel('日期')
    plt.ylabel('销售额')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 产品类别销售额分布图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='category', y='category_sales', data=category_sales_pd)
    plt.title('产品类别销售额分布', fontsize=15)
    plt.xlabel('产品类别')
    plt.ylabel('销售额')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图表
    plt.show()


# 执行Spark批处理分析流程
def run_spark_analysis():
    print("初始化Spark批处理分析流程...")

    # 创建Spark会话
    # spark = create_spark_session()

    # 定义数据模式
    # order_schema, user_schema, product_schema = define_schema()

    # 加载数据
    # orders_df = load_data(spark, order_schema)

    # 执行销售分析
    # daily_sales, category_sales, user_spending = sales_analysis(spark, orders_df)

    # 执行RFM客户价值分析
    # rfm_results, customer_segments = rfm_analysis(spark)

    # 执行商品关联分析
    # basket_analysis = product_association(spark)

    # 可视化分析结果
    # visualize_results(daily_sales, category_sales)

    print("Spark批处理分析流程示例完成")


# 显示Spark批处理分析流程
print("Spark批处理计算模块 - 分析流程")
print("1. 创建Spark会话并配置资源")
print("2. 定义数据模式")
print("3. 从HDFS加载订单、用户、产品数据")
print("4. 执行销售分析（每日销售额、产品类别销售、用户消费等级）")
print("5. 执行RFM客户价值分析")
print("6. 执行商品关联分析")
print("7. 可视化分析结果")

# 运行示例（注释掉以防止意外执行）
# run_spark_analysis()