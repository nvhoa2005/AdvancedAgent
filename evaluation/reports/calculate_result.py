import pandas as pd

chart = pd.read_csv("chart_report.csv")
edge = pd.read_csv("edge_cases_report.csv")
multihop = pd.read_csv("multihop_report.csv")
sql = pd.read_csv("sql_report.csv")
ragas = pd.read_csv("rag_report_ragas.csv")

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.upper()
    return df

chart = normalize_columns(chart)
edge = normalize_columns(edge)
multihop = normalize_columns(multihop)
sql = normalize_columns(sql)

def compute_accuracy(df):
    total = len(df)
    passed = (df["STATUS"] == "PASS").sum()
    accuracy = passed / total
    return passed, total, accuracy

acc_chart = compute_accuracy(chart)
acc_edge = compute_accuracy(edge)
acc_multihop = compute_accuracy(multihop)
acc_sql = compute_accuracy(sql)

metrics = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
]

ragas_mean = ragas[metrics].mean()

report = f"""
================ EVALUATION REPORT ================

1. Accuracy (PASS / TOTAL)

Chart Task       : {acc_chart[0]}/{acc_chart[1]}  | Accuracy = {acc_chart[2]:.4f} ({acc_chart[2]*100:.2f}%)
Edge Cases Task  : {acc_edge[0]}/{acc_edge[1]}  | Accuracy = {acc_edge[2]:.4f} ({acc_edge[2]*100:.2f}%)
Multihop Task    : {acc_multihop[0]}/{acc_multihop[1]}  | Accuracy = {acc_multihop[2]:.4f} ({acc_multihop[2]*100:.2f}%)
SQL Task         : {acc_sql[0]}/{acc_sql[1]}  | Accuracy = {acc_sql[2]:.4f} ({acc_sql[2]*100:.2f}%)

---------------------------------------------------

2. RAGAS Metrics For RAG (Average)

Faithfulness       : {ragas_mean['faithfulness']:.4f}
Answer Relevancy   : {ragas_mean['answer_relevancy']:.4f}
Context Precision  : {ragas_mean['context_precision']:.4f}
Context Recall     : {ragas_mean['context_recall']:.4f}

---------------------------------------------------

Total RAGAS Samples: {len(ragas)}
Total Evaluation Samples: {len(chart) + len(edge) + len(multihop) + len(sql) + len(ragas)}

===================================================
"""

with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)