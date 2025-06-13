
import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import os


# 模拟创建样本CSV数据
def create_sample_csv(filename="historical_orders.csv", records=1000):
    np.random.seed(42)  # 确保结果可重现

    # 生成随机数据
    order_ids = [f"ORD-HIST-{i}" for i in range(1, records + 1)]
    user_ids = [f"USER-{np.random.randint(1, 500)}" for _ in range(records)]
    product_ids = [f"PROD-{np.random.randint(1, 200)}" for _ in range(records)]
    quantities = np.random.randint(1, 10, size=records)
    prices = np.round(np.random.uniform(10, 1000, size=records), 2)
    total_amounts = np.round(quantities * prices, 2)

    # 生成过去30天的随机日期
    import datetime as dt
    today = dt.datetime.now()
    dates = [(today - dt.timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
             for _ in range(records)]

    # 状态
    statuses = np.random.choice(["completed", "returned", "cancelled"], size=records,
                                p=[0.85, 0.1, 0.05])

    # 创建数据框
    df = pd.DataFrame({
        "order_id": order_ids,
        "user_id": user_ids,
        "product_id": product_ids,
        "quantity": quantities,
        "price": prices,
        "total_amount": total_amounts,
        "timestamp": dates,
        "status": statuses
    })

    # 保存为CSV
    df.to_csv(filename, index=False)
    print(f"已创建样本CSV文件：{filename}，共{records}条记录")
    return df


# 从CSV导入数据到MySQL
def import_csv_to_mysql(csv_file, table_name, connection_string):
    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        print(f"从CSV文件读取了{len(df)}条记录")

        # 连接到MySQL
        engine = create_engine(connection_string)

        # 写入MySQL
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"已成功导入数据到MySQL表：{table_name}")

        return True
    except Exception as e:
        print(f"导入过程中出错：{str(e)}")
        return False


# 创建示例CSV数据
sample_df = create_sample_csv(records=20)
print(sample_df.head())

# MySQL连接配置 (仅示例，实际环境中请适当修改)
# connection_str = "mysql+mysqlconnector://username:password@localhost/ecommerce_db"
# import_csv_to_mysql("historical_orders.csv", "historical_orders", connection_str)