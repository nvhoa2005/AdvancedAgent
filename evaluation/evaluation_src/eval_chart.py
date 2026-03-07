import json
import uuid
import sys
import os
import csv
import shutil 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
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
    project_root = os.path.abspath(os.path.join(base_dir, '../..'))
    
    data_path = os.path.join(base_dir, '../ground_truth/chart_ground_truth.json')
    report_dir = os.path.join(base_dir, '../reports')
    
    chart_images_dir = os.path.join(report_dir, 'chart_reports')
    os.makedirs(chart_images_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, 'chart_report.csv')
    
    static_chart_path = os.path.join(project_root, 'static', 'chart_output.png')
    
    test_cases = load_ground_truth(data_path)
    total_cases = len(test_cases)
    passed_cases = 0
    report_data = []

    print(f"\n{YELLOW}BẮT ĐẦU ĐÁNH GIÁ PYTHON CHART MAKER VÀ LƯU ẢNH...{RESET}\n")

    for idx, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"chart_{idx}")
        question = case["question"]
        expected_tool = case["expected_tool"]
        complexity = case.get("complexity", "unknown")
        
        print(f"[{idx}/{total_cases}] Chạy Test: {case_id} ({complexity.upper()})")
        
        if os.path.exists(static_chart_path):
            try:
                os.remove(static_chart_path)
            except Exception as e:
                print(f"Cảnh báo: Không thể xóa ảnh cũ {static_chart_path}: {e}")

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        eval_status = "FAIL"
        eval_reason = ""
        agent_code = "NONE"
        tool_output_str = "NONE"
        saved_image_path = "NONE"
        
        try:
            result_state = agent_app.invoke(initial_state, config=config)
            messages = result_state.get("messages", [])
            
            called_python_tool = False
            
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc['name'] == expected_tool:
                            called_python_tool = True
                            agent_code = tc['args'].get('code', 'Không tìm thấy code')
                
                if isinstance(msg, ToolMessage) and msg.name == expected_tool:
                    tool_output_str = msg.content
            
            if not called_python_tool:
                eval_reason = f"Agent KHÔNG gọi tool {expected_tool}."
            else:
                if "Đã vẽ biểu đồ thành công" in tool_output_str:
                    if os.path.exists(static_chart_path):
                        target_image_name = f"{case_id}.png"
                        target_image_path = os.path.join(chart_images_dir, target_image_name)
                        
                        shutil.copy(static_chart_path, target_image_path)
                        
                        saved_image_path = f"chart_reports/{target_image_name}"
                        
                        eval_status = "PASS"
                        eval_reason = "Vẽ và lưu biểu đồ thành công."
                        passed_cases += 1
                    else:
                        eval_reason = "Tool báo thành công nhưng không tìm thấy file ảnh vật lý."
                
                elif "Vẽ biểu đồ không thành công" in tool_output_str:
                    eval_reason = "Code Python chạy được nhưng không có biểu đồ nào được vẽ ra."
                elif "Lỗi Python:" in tool_output_str:
                    short_error = tool_output_str.split("Lỗi Python:")[1][:150].strip()
                    eval_reason = f"Lỗi Runtime: {short_error}..."
                else:
                    eval_reason = f"Lỗi không xác định: {tool_output_str[:100]}..."

        except Exception as e:
            eval_reason = f"Lỗi System/LangGraph: {str(e)}"
            eval_status = "ERROR"

        if eval_status == "PASS":
            print(f"{GREEN}PASS{RESET} -> Đã lưu ảnh: {saved_image_path}")
        else:
            print(f"{RED}{eval_status}{RESET} - Lý do: {eval_reason}")

        report_data.append({
            "ID": case_id,
            "Complexity": complexity,
            "Question": question,
            "Status": eval_status,
            "Reason": eval_reason,
            "Image_Path": saved_image_path,
            "Agent_Code": agent_code.replace('\n', ' | '),
            "Tool_Output": tool_output_str.replace('\n', ' ')
        })

    print(f"\n{YELLOW}Đang lưu báo cáo Chart ra file CSV...{RESET}")
    headers = ["ID", "Complexity", "Question", "Status", "Reason", "Image_Path", "Agent_Code", "Tool_Output"]
    
    with open(report_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(report_data)

    accuracy = (passed_cases / total_cases) * 100
    print(f"\n{YELLOW}TỔNG KẾT PYTHON CHART MAKER:{RESET}")
    print(f"File Report đã lưu tại: {report_file}")
    print(f"Thư mục chứa ảnh: {chart_images_dir}")
    print(f"Độ chính xác: {GREEN if accuracy >= 70 else RED}{accuracy:.2f}%{RESET}\n")

if __name__ == "__main__":
    run_eval_pipeline()