import json
import uuid
import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.messages import HumanMessage, AIMessage
from app import init_agent_app

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def load_ground_truth(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_eval_pipeline():
    print(f"{YELLOW}Đang khởi tạo Insight Agent App...{RESET}")
    agent_app = init_agent_app()
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '../ground_truth/multihop.json')
    report_dir = os.path.join(base_dir, '../reports')
    
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, 'multihop_report.csv')
    
    test_cases = load_ground_truth(data_path)
    total_cases = len(test_cases)
    passed_cases = 0
    report_data = []

    print(f"\n{YELLOW}BẮT ĐẦU ĐÁNH GIÁ MULTI-HOP (TƯ DUY ĐA BƯỚC)...{RESET}\n")

    for idx, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"multihop_{idx}")
        question = case["question"]
        expected_tools = case["expected_tool"]
        
        if isinstance(expected_tools, str):
            expected_tools = [expected_tools]
            
        complexity = case.get("complexity", "expert")
        
        print(f"[{idx}/{total_cases}] Chạy Test: {case_id}")
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        eval_status = "FAIL"
        eval_reason = ""
        tools_called_in_order = []
        retry_count = 0
        final_message = ""
        
        try:
            result_state = agent_app.invoke(initial_state, config=config)
            messages = result_state.get("messages", [])
            retry_count = result_state.get("retry_count", 0)
            
            if messages:
                final_message = messages[-1].content
            
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_called_in_order.append(tc['name'])
            
            missing_tools = [t for t in expected_tools if t not in tools_called_in_order]
            
            if not missing_tools:
                eval_status = "PASS"
                passed_cases += 1
                eval_reason = f"Đã gọi đủ chuỗi Tools. Retry: {retry_count} lần."
            else:
                eval_reason = f"Thiếu tư duy gọi tool: {missing_tools}. Đã gọi: {tools_called_in_order}."
                
            if "lỗi" in case.get("evaluation_criteria", "").lower() and retry_count == 0:
                eval_status = "FAIL"
                eval_reason = "Không phát hiện thấy quá trình tự sửa lỗi (Retry = 0) dù câu hỏi có bẫy."
                if eval_status == "PASS": 
                    passed_cases -= 1

        except Exception as e:
            eval_reason = f"Lỗi System/LangGraph: {str(e)}"
            eval_status = "ERROR"

        if eval_status == "PASS":
            print(f"{GREEN}PASS{RESET} (Tools called: {tools_called_in_order} | Retries: {retry_count})")
        else:
            print(f"{RED}{eval_status}{RESET} - Lý do: {eval_reason}")

        report_data.append({
            "ID": case_id,
            "Complexity": complexity,
            "Question": question,
            "Expected_Tools": " -> ".join(expected_tools),
            "Actual_Tools_Called": " -> ".join(tools_called_in_order) if tools_called_in_order else "NONE",
            "Retry_Count": retry_count,
            "Status": eval_status,
            "Reason": eval_reason,
            "Final_Answer": final_message.replace('\n', ' ')
        })

    print(f"\n{YELLOW}Đang lưu báo cáo Multi-hop ra file CSV...{RESET}")
    headers = ["ID", "Complexity", "Question", "Expected_Tools", "Actual_Tools_Called", "Retry_Count", "Status", "Reason", "Final_Answer"]
    
    with open(report_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(report_data)

    accuracy = (passed_cases / total_cases) * 100
    print(f"\n{YELLOW}TỔNG KẾT MULTI-HOP (TOOL CHAINING):{RESET}")
    print(f"File Report đã lưu tại: {report_file}")
    print(f"Độ chính xác: {GREEN if accuracy >= 60 else RED}{accuracy:.2f}%{RESET}\n")

if __name__ == "__main__":
    run_eval_pipeline()