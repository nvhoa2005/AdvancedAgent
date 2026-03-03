import random
from faker import Faker
from sqlalchemy import create_engine, text
from config.settings import settings

class SQLDatabaseSeeder:
    """Class quản lý việc tạo Schema và sinh dữ liệu mẫu cho Database bán hàng."""
    
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        self.fake = Faker()

    def create_schema(self):
        with self.engine.connect() as conn:
            print("Đang xóa bảng cũ...")
            conn.execute(text("DROP TABLE IF EXISTS order_items CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS inventory CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS orders CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS customers CASCADE;"))
            
            conn.execute(text("""
                CREATE TABLE customers (
                    customer_id SERIAL PRIMARY KEY,
                    name VARCHAR(100), email VARCHAR(100), phone VARCHAR(50),
                    city VARCHAR(50), signup_date DATE
                );
            """))
            conn.execute(text("""
                CREATE TABLE products (
                    product_id SERIAL PRIMARY KEY,
                    name VARCHAR(100), category VARCHAR(50),
                    price DECIMAL(10, 2), cost DECIMAL(10, 2), supplier VARCHAR(100)
                );
            """))
            conn.execute(text("""
                CREATE TABLE inventory (
                    product_id INTEGER REFERENCES products(product_id),
                    stock_quantity INTEGER, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE orders (
                    order_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(customer_id),
                    order_date TIMESTAMP, status VARCHAR(20), total_amount DECIMAL(10, 2)
                );
            """))
            conn.execute(text("""
                CREATE TABLE order_items (
                    item_id SERIAL PRIMARY KEY,
                    order_id INTEGER REFERENCES orders(order_id),
                    product_id INTEGER REFERENCES products(product_id),
                    quantity INTEGER, unit_price DECIMAL(10, 2)
                );
            """))
            conn.commit()
        print("Đã tạo xong Schema 5 bảng.")

    def seed_data(self, num_customers=50, num_products=50, num_orders=200):
        categories_data = {
            'Electronics': ['iPhone 15', 'Samsung Galaxy', 'MacBook', 'Dell XPS'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers'],
            'Home': ['Coffee Maker', 'Blender', 'Desk Lamp', 'Sofa']
        }

        with self.engine.begin() as conn:
            print(f"Đang sinh {num_customers} khách hàng, {num_products} sản phẩm, {num_orders} đơn hàng...")

            customer_ids = []
            for _ in range(num_customers):
                res = conn.execute(
                    text("INSERT INTO customers (name, email, phone, city, signup_date) VALUES (:n, :e, :p, :c, :d) RETURNING customer_id"),
                    {"n": self.fake.name(), "e": self.fake.email(), "p": self.fake.phone_number(), "c": self.fake.city(), "d": self.fake.date_between(start_date='-2y', end_date='today')}
                )
                customer_ids.append(res.fetchone()[0])

            product_ids = []
            product_prices = {} 
            for _ in range(num_products):
                cat = random.choice(list(categories_data.keys()))
                base_name = random.choice(categories_data[cat])
                full_name = f"{base_name} {self.fake.word().capitalize()}"
                price = round(random.uniform(50, 2000), 2)
                
                res = conn.execute(
                    text("INSERT INTO products (name, category, price, cost, supplier) VALUES (:n, :c, :p, :co, :s) RETURNING product_id"),
                    {"n": full_name, "c": cat, "p": price, "co": round(price * 0.7, 2), "s": self.fake.company()}
                )
                pid = res.fetchone()[0]
                product_ids.append(pid)
                product_prices[pid] = price
                
                conn.execute(
                    text("INSERT INTO inventory (product_id, stock_quantity) VALUES (:pid, :qty)"),
                    {"pid": pid, "qty": random.randint(0, 100)}
                )

            for _ in range(num_orders):
                res = conn.execute(
                    text("INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES (:cid, :od, :s, 0) RETURNING order_id"),
                    {"cid": random.choice(customer_ids), "od": self.fake.date_time_between(start_date='-1y', end_date='now'), "s": random.choice(['Completed', 'Completed', 'Pending', 'Cancelled'])}
                )
                order_id = res.fetchone()[0]
                
                order_total = 0
                for pid in random.sample(product_ids, random.randint(1, 4)):
                    qty = random.randint(1, 3)
                    price = product_prices[pid]
                    order_total += price * qty
                    
                    conn.execute(
                        text("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (:oid, :pid, :q, :up)"),
                        {"oid": order_id, "pid": pid, "q": qty, "up": price}
                    )
                
                conn.execute(
                    text("UPDATE orders SET total_amount = :t WHERE order_id = :oid"),
                    {"t": order_total, "oid": order_id}
                )

        print("Đã hoàn tất seed data.")

    def run(self):
        """Hàm kích hoạt toàn bộ quy trình."""
        self.create_schema()
        self.seed_data()

if __name__ == "__main__":
    seeder = SQLDatabaseSeeder()
    seeder.run()