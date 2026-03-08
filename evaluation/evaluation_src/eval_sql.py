import json
import uuid
import sys
import os
import csv
import pandas as pd
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.messages import HumanMessage
from app import init_agent_app
from config.settings import settings

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

db_engine = create_engine(settings.DATABASE_URL)

def load_ground_truth(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_dataframes(df_expected, df_agent):
    """
    Hàm so sánh 2 DataFrame thông minh:
    1. Bỏ qua thứ tự dòng (sort toàn bộ dữ liệu).
    2. Bỏ qua tên cột (ép tên cột df_agent theo df_expected) vì LLM hay dùng AS alias.
    """
    if df_expected.shape != df_agent.shape:
        return False, f"Khác số lượng dòng/cột. Chuẩn: {df_expected.shape}, Agent: {df_agent.shape}"
    
    try:
        cols_expected = df_expected.columns.tolist()
        cols_agent = df_agent.columns.tolist()
        
        df_expected_sorted = df_expected.sort_values(by=cols_expected).reset_index(drop=True)
        df_agent_sorted = df_agent.sort_values(by=cols_agent).reset_index(drop=True)
        
        df_agent_sorted.columns = df_expected_sorted.columns
        
        is_match = df_expected_sorted.equals(df_agent_sorted)
        if is_match:
            return True, "Dữ liệu khớp hoàn toàn."
        else:
            return False, "Dữ liệu không khớp."
    except Exception as e:
        return False, f"Lỗi khi so sánh bảng: {str(e)}"

def run_eval_pipeline():
    print(f"{YELLOW}Đang khởi tạo Insight Agent App & Kết nối Database...{RESET}")
    agent_app = init_agent_app()
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '../ground_truth/sql_ground_truth.json')
    report_dir = os.path.join(base_dir, '../reports')
    
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, 'sql_report.csv')
    
    test_cases = load_ground_truth(data_path)
    total_cases = len(test_cases)
    passed_cases = 0
    report_data = []

    print(f"\n{YELLOW}BẮT ĐẦU ĐÁNH GIÁ DATA MATCH (SQL TEXT-TO-SQL)...{RESET}\n")

    for idx, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"sql_{idx}")
        question = case["question"]
        expected_sql = case["expected_sql"]
        complexity = case.get("complexity", "unknown")
        
        print(f"[{idx}/{total_cases}] Chạy Test: {case_id} ({complexity.upper()})")
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        agent_sql = None
        eval_status = "FAIL"
        eval_reason = ""
        final_message = ""
        
        try:
            result_state = agent_app.invoke(initial_state, config=config)
            
            if result_state.get("messages"):
                final_message = result_state["messages"][-1].content
            
            for msg in result_state["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc['name'] == 'query_sql_db':
                            agent_sql = tc['args'].get('query')
            
            if not agent_sql:
                eval_reason = "Agent KHÔNG gọi tool query_sql_db."
            else:
                try:
                    df_expected = pd.read_sql_query(expected_sql, db_engine)
                    df_agent = pd.read_sql_query(agent_sql, db_engine)
                    
                    is_match, reason = compare_dataframes(df_expected, df_agent)
                    
                    if is_match:
                        eval_status = "PASS"
                        passed_cases += 1
                        eval_reason = "Khớp dữ liệu (Exact Match)"
                    else:
                        eval_reason = f"Sai dữ liệu: {reason}"
                        
                except Exception as db_err:
                    eval_reason = f"Lỗi thực thi SQL của Agent: {str(db_err)[:100]}..."

        except Exception as e:
            eval_reason = f"Lỗi System/LangGraph: {str(e)}"
            eval_status = "ERROR"

        if eval_status == "PASS":
            print(f"{GREEN}PASS{RESET}")
        else:
            print(f"{RED}{eval_status}{RESET} - Lý do: {eval_reason}")
            print(f"   Expected SQL: {expected_sql}")
            print(f"   Agent SQL: {agent_sql}")

        report_data.append({
            "ID": case_id,
            "Complexity": complexity,
            "Question": question,
            "Expected_SQL": expected_sql,
            "Agent_SQL": agent_sql if agent_sql else "NONE",
            "Status": eval_status,
            "Reason": eval_reason,
            "Agent_Response": final_message.replace('\n', ' ')
        })

    print(f"\n{YELLOW}Đang lưu báo cáo SQL ra file CSV...{RESET}")
    headers = ["ID", "Complexity", "Question", "Expected_SQL", "Agent_SQL", "Status", "Reason", "Agent_Response"]
    
    with open(report_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(report_data)

    accuracy = (passed_cases / total_cases) * 100
    print(f"\n{YELLOW}TỔNG KẾT DATA MATCH (SQL):{RESET}")
    print(f"File Report đã lưu tại: {report_file}")
    print(f"Độ chính xác: {GREEN if accuracy >= 70 else RED}{accuracy:.2f}%{RESET}\n")

if __name__ == "__main__":
    run_eval_pipeline()