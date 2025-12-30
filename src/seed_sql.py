import os
import random
from faker import Faker
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Kết nối DB
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("Chưa tìm thấy DATABASE_URL trong file .env")

engine = create_engine(db_url)
fake = Faker()

def create_schema():
    with engine.connect() as conn:
        
        print("Đang xóa bảng cũ...")
        conn.execute(text("DROP TABLE IF EXISTS order_items CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS inventory CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS orders CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS customers CASCADE;"))
        
        # 1. Bảng Customers
        conn.execute(text("""
            CREATE TABLE customers (
                customer_id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100),
                phone VARCHAR(50),
                city VARCHAR(50),
                signup_date DATE
            );
        """))

        # 2. Bảng Products
        conn.execute(text("""
            CREATE TABLE products (
                product_id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                category VARCHAR(50),
                price DECIMAL(10, 2),
                cost DECIMAL(10, 2),
                supplier VARCHAR(100)
            );
        """))

        # 3. Bảng Inventory
        conn.execute(text("""
            CREATE TABLE inventory (
                product_id INTEGER REFERENCES products(product_id),
                stock_quantity INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # 4. Bảng Orders
        conn.execute(text("""
            CREATE TABLE orders (
                order_id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(customer_id),
                order_date TIMESTAMP,
                status VARCHAR(20),
                total_amount DECIMAL(10, 2)
            );
        """))

        # 5. Bảng Order Items
        conn.execute(text("""
            CREATE TABLE order_items (
                item_id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES orders(order_id),
                product_id INTEGER REFERENCES products(product_id),
                quantity INTEGER,
                unit_price DECIMAL(10, 2)
            );
        """))
        
        conn.commit()
    print("Đã tạo xong Schema 5 bảng.")

def seed_data():
    # Cấu hình số lượng dữ liệu
    NUM_CUSTOMERS = 50
    NUM_PRODUCTS = 50  
    NUM_ORDERS = 200

    # Danh sách mẫu để random
    categories_data = {
        'Electronics': ['iPhone 15', 'Samsung Galaxy', 'MacBook', 'Dell XPS', 'Sony Headphones', 'LG Monitor', 'iPad Pro'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hat', 'Scarf', 'Hoodie'],
        'Home': ['Coffee Maker', 'Blender', 'Desk Lamp', 'Air Purifier', 'Smart Lock', 'Vacuum', 'Sofa']
    }

    with engine.begin() as conn:
        print("Đang sinh dữ liệu...")

        # --- 1. SEED CUSTOMERS ---
        customer_ids = []
        for _ in range(NUM_CUSTOMERS):
            res = conn.execute(
                text("INSERT INTO customers (name, email, phone, city, signup_date) VALUES (:n, :e, :p, :c, :d) RETURNING customer_id"),
                {
                    "n": fake.name(),
                    "e": fake.email(),
                    "p": fake.phone_number(),
                    "c": fake.city(),
                    "d": fake.date_between(start_date='-2y', end_date='today')
                }
            )
            customer_ids.append(res.fetchone()[0])

        # --- 2. SEED PRODUCTS ---
        product_ids = []
        product_prices = {} 
        
        category_keys = list(categories_data.keys()) 
        
        for _ in range(NUM_PRODUCTS):
            cat = random.choice(category_keys)
            base_name = random.choice(categories_data[cat])
            full_name = f"{base_name} {fake.word().capitalize()}"
            
            price = round(random.uniform(50, 2000), 2)
            cost = round(price * 0.7, 2)
            
            res = conn.execute(
                text("INSERT INTO products (name, category, price, cost, supplier) VALUES (:n, :c, :p, :co, :s) RETURNING product_id"),
                {
                    "n": full_name, "c": cat, "p": price, "co": cost, "s": fake.company()
                }
            )
            pid = res.fetchone()[0]
            product_ids.append(pid)
            product_prices[pid] = price
            
            # Tạo Inventory
            conn.execute(
                text("INSERT INTO inventory (product_id, stock_quantity) VALUES (:pid, :qty)"),
                {"pid": pid, "qty": random.randint(0, 100)}
            )

        # --- 3. SEED ORDERS & ITEMS ---
        statuses = ['Completed', 'Completed', 'Completed', 'Pending', 'Cancelled']
        
        for _ in range(NUM_ORDERS):
            cust_id = random.choice(customer_ids)
            order_date = fake.date_time_between(start_date='-1y', end_date='now')
            status = random.choice(statuses)
            
            res = conn.execute(
                text("INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES (:cid, :od, :s, 0) RETURNING order_id"),
                {"cid": cust_id, "od": order_date, "s": status}
            )
            order_id = res.fetchone()[0]
            
            num_items = random.randint(1, 4)
            chosen_products = random.sample(product_ids, num_items)
            order_total = 0
            
            for pid in chosen_products:
                qty = random.randint(1, 3)
                price = product_prices[pid]
                line_total = price * qty
                order_total += line_total
                
                conn.execute(
                    text("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (:oid, :pid, :q, :up)"),
                    {"oid": order_id, "pid": pid, "q": qty, "up": price}
                )
            
            conn.execute(
                text("UPDATE orders SET total_amount = :t WHERE order_id = :oid"),
                {"t": order_total, "oid": order_id}
            )

    print("Đã hoàn tất data")

if __name__ == "__main__":
    create_schema()
    seed_data()