from src.tools import query_sql_db, search_policy_docs, python_chart_maker, get_db_schema

print("=== 1. TEST SCHEMA ===")
print(get_db_schema()[:300] + "...\n") 

print("=== 2. TEST SQL QUERY ===")
sql = "SELECT count(*) FROM orders"
print(f"Run SQL: {sql}")
print(f"Result: {query_sql_db.invoke(sql)}\n")

print("=== 3. TEST RAG SEARCH ===")
query = "Chế độ nghỉ phép"
print(f"Search: {query}")
res = search_policy_docs.invoke(query)
print(f"Result (First 100 chars): {res[:100]}...\n")

print("=== 4. TEST CHART GENERATION ===")
code = """
import matplotlib.pyplot as plt
data = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
plt.bar(labels, data)
plt.title("Test Chart")
"""
print("Generating chart...")
print(python_chart_maker.invoke(code))