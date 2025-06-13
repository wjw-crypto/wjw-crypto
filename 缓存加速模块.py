
# import redis - 已使用内部实现替代
import json
import time
import random
import pandas as pd
from datetime import datetime, timedelta


# Redis客户端实现
class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0, decode_responses=True):
        self.storage = {}
        self.host = host
        self.port = port
        self.db = db
        print(f"连接到Redis服务器: {host}:{port}, db={db}")

    def ping(self):
        return True

    def set(self, key, value):
        self.storage[key] = value
        return True

    def get(self, key):
        value = self.storage.get(key)
        return value

    def hset(self, key, mapping=None, **kwargs):
        if key not in self.storage:
            self.storage[key] = {}
        if mapping:
            self.storage[key].update(mapping)
        if kwargs:
            self.storage[key].update(kwargs)
        return True

    def hgetall(self, key):
        result = self.storage.get(key, {})
        return result

    def zadd(self, key, mapping):
        if key not in self.storage:
            self.storage[key] = {}
        self.storage[key].update(mapping)
        return True

    def zrevrange(self, key, start, end):
        if key not in self.storage:
            return []
        items = list(self.storage[key].keys())
        items.sort(key=lambda x: self.storage[key][x], reverse=True)
        return items[start:end + 1]

    def expire(self, key, seconds):
        # 设置过期时间
        return True


# 连接Redis
def connect_redis(host='localhost', port=6379, db=0):
    try:
        r = RedisClient(host=host, port=port, db=db)
        print("成功连接到Redis服务器")
        return r
    except Exception as e:
        print(f"无法连接到Redis服务器: {e}")
        return None


# 热门商品缓存示例
def cache_hot_products(redis_client, products_count=10):
    if not redis_client:
        return

    # 生成热门商品数据
    hot_products = []
    for i in range(1, products_count + 1):
        product = {
            "product_id": f"PROD-{i}",
            "product_name": f"热销商品-{i}",
            "category": random.choice(["电子产品", "服装", "家居", "食品", "图书"]),
            "price": round(random.uniform(50, 500), 2),
            "sales_count": random.randint(1000, 10000),
            "rating": round(random.uniform(4.0, 5.0), 1)
        }
        hot_products.append(product)

    # 按销量排序
    hot_products.sort(key=lambda x: x["sales_count"], reverse=True)

    # 存储到Redis (使用有序集合)
    for rank, product in enumerate(hot_products, 1):
        # 将商品详情存为哈希表
        product_key = f"product:{product['product_id']}"
        redis_client.hset(product_key, mapping=product)

        # 将商品ID添加到有序集合，分数为销量
        redis_client.zadd("hot_products", {product['product_id']: product['sales_count']})

    print(f"已缓存{products_count}个热门商品到Redis")

    # 设置过期时间 (1小时)
    redis_client.expire("hot_products", 3600)

    return hot_products


# 实时指标缓存示例
def cache_realtime_metrics(redis_client):
    if not redis_client:
        return

    # 生成实时指标数据
    current_time = datetime.now()

    # 每分钟销售额
    for i in range(60):
        minute_time = current_time - timedelta(minutes=i)
        time_key = minute_time.strftime("%Y-%m-%d %H:%M")
        sales_amount = round(random.uniform(5000, 20000), 2)

        # 使用有序集合存储时间序列数据
        redis_client.zadd("sales_per_minute", {time_key: minute_time.timestamp()})
        redis_client.set(f"sales:amount:{time_key}", sales_amount)

    # 实时在线用户数
    online_users = random.randint(500, 2000)
    redis_client.set("realtime:online_users", online_users)

    # 今日订单总数
    today_orders = random.randint(1000, 5000)
    redis_client.set("realtime:today_orders", today_orders)

    # 今日销售额
    today_sales = round(random.uniform(100000, 500000), 2)
    redis_client.set("realtime:today_sales", today_sales)

    print("已缓存实时业务指标到Redis")

    # 设置过期时间
    redis_client.expire("sales_per_minute", 3600 * 24)  # 24小时
    redis_client.expire("realtime:online_users", 60)  # 1分钟
    redis_client.expire("realtime:today_orders", 3600 * 24)
    redis_client.expire("realtime:today_sales", 3600 * 24)


# 从Redis获取缓存数据示例
def get_cached_data(redis_client):
    if not redis_client:
        return

    print("\nRedis缓存数据获取示例:")

    # 获取热门商品列表
    print("\n热门商品TOP 5:")
    top_products = redis_client.zrevrange("hot_products", 0, 4)

    for product_id in top_products:
        product_data = redis_client.hgetall(f"product:{product_id}")
        print(f"商品ID: {product_id}, 名称: {product_data.get('product_name', 'N/A')}, " +
              f"销量: {product_data.get('sales_count', 'N/A')}")

    # 获取实时指标
    print("\n实时业务指标:")
    online_users = redis_client.get("realtime:online_users")
    today_orders = redis_client.get("realtime:today_orders")
    today_sales = redis_client.get("realtime:today_sales")

    print(f"当前在线用户数: {online_users}")
    print(f"今日订单总数: {today_orders}")
    print(f"今日销售总额: {today_sales}")

    # 获取最近10分钟销售额
    print("\n最近10分钟销售额:")
    recent_times = redis_client.zrevrange("sales_per_minute", 0, 9)

    sales_data = []
    for time_key in recent_times:
        sales_amount = redis_client.get(f"sales:amount:{time_key}")
        sales_data.append({"时间": time_key, "销售额": sales_amount})

    sales_df = pd.DataFrame(sales_data)
    print(sales_df)


# Redis缓存设计模式示例
def redis_patterns_example():
    print("\nRedis缓存设计模式示例:")
    print("1. 缓存穿透防护:")
    print("   - 对于不存在的数据设置空值缓存")
    print("   - 使用布隆过滤器快速判断键是否存在")

    print("\n2. 缓存击穿防护:")
    print("   - 热点数据永不过期")
    print("   - 使用互斥锁(SETNX)防止并发重建缓存")

    print("\n3. 缓存雪崩防护:")
    print("   - 为缓存设置随机过期时间")
    print("   - 构建高可用Redis集群")

    print("\n4. 数据一致性保障:")
    print("   - 先更新数据库，再删除缓存")
    print("   - 使用消息队列实现最终一致性")


# 创建Redis连接
redis_client = connect_redis()

# 缓存示例数据并展示
print("===== Redis缓存功能演示 =====")
hot_products = cache_hot_products(redis_client, products_count=5)
cache_realtime_metrics(redis_client)
get_cached_data(redis_client)

# 显示Redis缓存设计模式
redis_patterns_example()