import json
import uuid
import sys
import os
import pandas as pd
from config.settings import settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app import init_agent_app
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_eval_pipeline():
    USE_CACHE = True 

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '../ground_truth/rag_ground_truth.json')
    report_dir = os.path.join(base_dir, '../reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, 'rag_report_ragas.csv')
    
    intermediate_file = os.path.join(report_dir, 'ragas_intermediate_cache.json')
    
    ragas_data = {
        "question": [], "answer": [], "contexts": [], "ground_truth": []
    }
    meta_data = []

    if USE_CACHE and os.path.exists(intermediate_file):
        print(f"\n{YELLOW}ĐANG SỬ DỤNG DỮ LIỆU CACHE TRUNG GIAN (BỎ QUA BƯỚC 1)...{RESET}")
        cached_data = load_json(intermediate_file)
        ragas_data = cached_data["ragas_data"]
        meta_data = cached_data["meta_data"]
        print(f"Đã load {len(ragas_data['question'])} record từ file cache.")
    
    else:
        print(f"{YELLOW}Đang khởi tạo Insight Agent App...{RESET}")
        agent_app = init_agent_app()
        test_cases = load_json(data_path)
        total_cases = len(test_cases)

        print(f"\n{YELLOW}BƯỚC 1/2: CHẠY AGENT ĐỂ THU THẬP TRACES (RAG)...{RESET}\n")

        for idx, case in enumerate(test_cases, 1):
            case_id = case.get("id", f"rag_{idx}")
            question = case["question"]
            expected_tool = case.get("expected_tool", "search_policy_docs")
            ground_truth = case["ground_truth"]
            
            print(f"[{idx}/{total_cases}] Đang chạy Agent: {case_id}")
            
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            initial_state = {"messages": [HumanMessage(content=question)]}
            
            agent_answer = "ERROR"
            retrieved_contexts = []
            
            try:
                result_state = agent_app.invoke(initial_state, config=config)
                messages = result_state.get("messages", [])
                
                if len(messages) >= 2:
                    agent_answer = messages[-2].content
                elif messages:
                    agent_answer = messages[-1].content
                agent_answer = agent_answer.strip().strip('"').strip("'")
                
                for msg in messages:
                    if isinstance(msg, ToolMessage) and msg.name == expected_tool:
                        retrieved_contexts.append(msg.content)
                
                if not retrieved_contexts:
                    retrieved_contexts = ["Không có ngữ cảnh nào được truy xuất."]

            except Exception as e:
                agent_answer = f"Lỗi System: {str(e)}"
                retrieved_contexts = ["ERROR"]
                print(f"   {RED}Lỗi chạy Agent: {e}{RESET}")

            ragas_data["question"].append(question)
            ragas_data["answer"].append(agent_answer)
            ragas_data["contexts"].append(retrieved_contexts) 
            ragas_data["ground_truth"].append(ground_truth)
            meta_data.append({"ID": case_id, "Complexity": case.get("complexity", "")})

        print(f"\n{YELLOW}Đang lưu dữ liệu trung gian ra file JSON...{RESET}")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump({
                "ragas_data": ragas_data,
                "meta_data": meta_data
            }, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu cache tại: {intermediate_file}")


    print(f"\n{YELLOW}BƯỚC 2/2: KHỞI ĐỘNG RAGAS LLM-AS-A-JUDGE...{RESET}")
    
    try:
        dataset = Dataset.from_dict(ragas_data)
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        evaluator_llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
        evaluator_embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        
        result = evaluate(
            dataset, 
            metrics=metrics,
            llm=evaluator_llm,               
            embeddings=evaluator_embeddings  
        )
        df_result = result.to_pandas()
        
        df_meta = pd.DataFrame(meta_data)
        df_final = pd.concat([df_meta, df_result], axis=1)
        
        df_final.to_csv(report_file, index=False, encoding='utf-8-sig')

        print(f"\n{YELLOW}TỔNG KẾT RAGAS XONG{RESET}")

    except Exception as e:
        print(f"\n{RED}BƯỚC 2 GẶP LỖI: {str(e)}{RESET}")

if __name__ == "__main__":
    run_eval_pipeline()