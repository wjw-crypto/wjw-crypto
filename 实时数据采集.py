from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

# 配置Kafka生产者
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


# 模拟生成实时订单数据
def generate_order():
    order_id = f"ORD-{int(time.time())}-{random.randint(1000, 9999)}"
    user_id = f"USER-{random.randint(1, 1000)}"
    product_id = f"PROD-{random.randint(1, 500)}"
    quantity = random.randint(1, 5)
    price = round(random.uniform(10, 1000), 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "order_id": order_id,
        "user_id": user_id,
        "product_id": product_id,
        "quantity": quantity,
        "price": price,
        "total_amount": round(quantity * price, 2),
        "timestamp": timestamp,
        "status": "created"
    }


# 发送数据到Kafka主题
def send_to_kafka(topic_name="ecommerce-orders", records_count=5):
    for _ in range(records_count):
        order_data = generate_order()
        producer.send(topic_name, order_data)
        print(f"已发送订单数据: {order_data}")
        time.sleep(0.5)  # 模拟每0.5秒产生一笔订单


# 执行发送操作
# send_to_kafka(records_count=5)  # 注释掉以防止意外执行
print("实时数据采集模块准备就绪，取消注释最后一行代码可发送模拟订单数据")