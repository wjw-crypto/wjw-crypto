
from kafka import KafkaConsumer
import json
import threading
import time


# 配置Kafka消费者
def create_kafka_consumer(topic_name, group_id="ecommerce-analytics-group"):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    return consumer


# 实时消费Kafka数据
def consume_orders(topic_name="ecommerce-orders"):
    consumer = create_kafka_consumer(topic_name)
    print(f"开始消费Kafka主题 '{topic_name}' 的数据...")

    try:
        for message in consumer:
            order_data = message.value
            print(f"收到订单数据: {order_data}")

            # 这里可以添加数据处理逻辑
            # 例如：将数据转发到不同的目的地（HDFS、Redis、MySQL等）

    except KeyboardInterrupt:
        print("消费者已停止")
    finally:
        consumer.close()


# 模拟Kafka主题管理
def kafka_admin_operations():
    print("Kafka主题管理操作示例:")
    print(
        "1. 创建主题: kafka-topics.sh --create --topic ecommerce-orders --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1")
    print("2. 查看主题列表: kafka-topics.sh --list --bootstrap-server localhost:9092")
    print("3. 查看主题详情: kafka-topics.sh --describe --topic ecommerce-orders --bootstrap-server localhost:9092")
    print("4. 查看消费组: kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list")
    print(
        "5. 查看消费组详情: kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group ecommerce-analytics-group")


# 在后台线程中运行消费者（仅作示例）
def run_consumer_in_background():
    consumer_thread = threading.Thread(target=consume_orders)
    consumer_thread.daemon = True
    consumer_thread.start()
    print("消费者已在后台启动，将持续监听订单数据...")
    return consumer_thread


# 显示Kafka管理操作
kafka_admin_operations()

# 注释掉以下代码以防止意外执行
# consumer_thread = run_consumer_in_background()
# time.sleep(60)  # 让消费者运行60秒