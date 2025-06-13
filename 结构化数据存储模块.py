
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# MySQL数据库模式设计
Base = declarative_base()


# 定义数据模型
class User(Base):
    __tablename__ = 'users'

    user_id = Column(String(50), primary_key=True)
    username = Column(String(100))
    email = Column(String(100))
    registration_date = Column(DateTime)
    last_login = Column(DateTime)
    user_level = Column(String(20))

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"


class Product(Base):
    __tablename__ = 'products'

    product_id = Column(String(50), primary_key=True)
    product_name = Column(String(200))
    category = Column(String(100))
    price = Column(Float)
    inventory = Column(Integer)

    def __repr__(self):
        return f"<Product(product_id='{self.product_id}', product_name='{self.product_name}')>"


class Order(Base):
    __tablename__ = 'orders'

    order_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'))
    order_date = Column(DateTime)
    total_amount = Column(Float)
    status = Column(String(20))

    def __repr__(self):
        return f"<Order(order_id='{self.order_id}', user_id='{self.user_id}', total_amount={self.total_amount})>"


class OrderItem(Base):
    __tablename__ = 'order_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), ForeignKey('orders.order_id'))
    product_id = Column(String(50), ForeignKey('products.product_id'))
    quantity = Column(Integer)
    price = Column(Float)

    def __repr__(self):
        return f"<OrderItem(order_id='{self.order_id}', product_id='{self.product_id}', quantity={self.quantity})>"


# 创建数据库和表
def setup_database(connection_string):
    try:
        # 创建引擎和表
        engine = create_engine(connection_string)
        Base.metadata.create_all(engine)
        print("数据库表已成功创建")
        return engine
    except Exception as e:
        print(f"创建数据库表时出错：{str(e)}")
        return None


# 生成示例数据
def generate_sample_data(engine, users_count=50, products_count=100):
    if not engine:
        print("数据库引擎未初始化")
        return

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 生成用户数据
        for i in range(1, users_count + 1):
            user = User(
                user_id=f"USER-{i}",
                username=f"user{i}",
                email=f"user{i}@example.com",
                registration_date=datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365)),
                last_login=datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30)),
                user_level=random.choice(["bronze", "silver", "gold", "platinum"])
            )
            session.add(user)

        # 生成产品数据
        categories = ["电子产品", "服装", "家居", "食品", "图书"]
        for i in range(1, products_count + 1):
            product = Product(
                product_id=f"PROD-{i}",
                product_name=f"产品-{i}",
                category=random.choice(categories),
                price=round(random.uniform(10, 1000), 2),
                inventory=random.randint(0, 1000)
            )
            session.add(product)

        session.commit()
        print(f"已生成{users_count}个用户和{products_count}个产品的示例数据")
    except Exception as e:
        session.rollback()
        print(f"生成示例数据时出错：{str(e)}")
    finally:
        session.close()


# 查询示例
def query_examples(engine):
    if not engine:
        print("数据库引擎未初始化")
        return

    try:
        # 使用pandas读取数据
        users_df = pd.read_sql("SELECT * FROM users LIMIT 5", engine)
        print("用户数据示例:")
        print(users_df)

        products_df = pd.read_sql("SELECT * FROM products LIMIT 5", engine)
        print("\n产品数据示例:")
        print(products_df)

        # 复杂查询示例
        query = """
        SELECT 
            o.order_id, 
            u.username, 
            SUM(oi.quantity * oi.price) as total_amount,
            o.order_date,
            o.status
        FROM 
            orders o
        JOIN 
            users u ON o.user_id = u.user_id
        JOIN 
            order_items oi ON o.order_id = oi.order_id
        GROUP BY 
            o.order_id, u.username, o.order_date, o.status
        LIMIT 5
        """

        print("\n复杂查询SQL示例:")
        print(query)

    except Exception as e:
        print(f"查询数据时出错：{str(e)}")


# 显示数据库操作示例
print("MySQL结构化数据存储模块 - 数据库设计")
print("表结构:")
print("1. users - 用户信息表")
print("2. products - 产品信息表")
print("3. orders - 订单主表")
print("4. order_items - 订单明细表")

# 注释掉以下代码以防止意外执行
# connection_str = "mysql+mysqlconnector://username:password@localhost/ecommerce_db"
# engine = setup_database(connection_str)
# if engine:
#     generate_sample_data(engine)
#     query_examples(engine