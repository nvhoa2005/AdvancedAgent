import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage

from src.graph import app as graph_app 

st.set_page_config(page_title="Advanced Agent", layout="wide")

async def run_chat_logic(user_input):
    with st.chat_message("assistant"):
        status_container = st.status("Đang xử lý...", expanded=True)
        answer_placeholder = st.empty()
        full_response = ""
        
        config = {"configurable": {"thread_id": "1"}}
        
        async for event in graph_app.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node", "")
            if kind == "on_tool_start":
                tool_name = event['name']
                if tool_name not in ["__start__", "__end__"]:
                    status_container.write(f"Đang dùng công cụ: **{tool_name}**...")
                    if tool_name == "query_sql_db":
                        sql_query = event.get("data", {}).get("input", {}).get("query")
                        if sql_query:
                            with status_container.expander("Xem câu lệnh SQL thực thi"):
                                st.code(sql_query, language="sql")
            elif kind == "on_tool_end":
                tool_name = event['name']
                if tool_name not in ["__start__", "__end__"]:
                    status_container.write(f"**{tool_name}** xong.")
                    if tool_name == "python_chart_maker":
                        st.image("static/chart_output.png", caption="Biểu đồ phân tích")
            elif kind == "on_chat_model_stream":
                if node_name in ["final_answer", "general_chat"]:
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        answer_placeholder.markdown(full_response + "▌")
            elif kind == "on_chain_end":
                if node_name == "input_guardrail" and event["name"] == "input_guardrail":
                    output = event["data"].get("output")
                    if output and not isinstance(output, str):
                        if isinstance(output, dict):
                            is_safe = output.get("is_safe", True)
                            reasoning = output.get("reasoning", "Không có lý do")
                        else:
                            is_safe = getattr(output, 'is_safe', True)
                            reasoning = getattr(output, 'reasoning', "Không có lý do")
                        label = "An toàn" if is_safe else "Cảnh báo Bảo mật"
                        color = "green" if is_safe else "red"
                        status_container.write(f"**Kiểm duyệt đầu vào:** Đã kiểm duyệt - :{color}[{label}]")
                        status_container.write(f"**Lý do:** {reasoning}")
                        status_container.write("---")
                elif node_name == "agent_router" and event["name"] == "agent_router":
                    output = event["data"].get("output")
                    if output and not isinstance(output, str):
                        try:
                            is_out = getattr(output, 'is_out_of_scope', None)
                            reasoning = getattr(output, 'reasoning', "Không có lý do")
                            
                            if is_out is None and isinstance(output, dict):
                                is_out = output.get("is_out_of_scope")
                                reasoning = output.get("reasoning", "Không có lý do")
                            if is_out is not None:
                                status_label = "Ngoài phạm vi" if is_out else "Trong phạm vi"
                                color = "orange" if is_out else "green"
                                
                                status_container.write(f"**Phân loại:** :{color}[{status_label}]")
                                status_container.write(f"**Lý do:** {reasoning}")
                                status_container.write("---")
                        except Exception as e:
                            print(f"Lỗi hiển thị log: {e}")
                elif node_name == "output_guardrail" and event["name"] == "output_guardrail":
                    status_container.write("**Kiểm duyệt đầu ra:** Dữ liệu nhạy cảm đã được lọc.")
                elif node_name == "query_transform" and event["name"] == "query_transform":
                    output = event["data"].get("output")
                    if output and isinstance(output, dict):
                        transformed = output.get("transformed_query")
                        if transformed:
                            status_container.write(f"**Tối ưu câu hỏi:** _{transformed}_")
                            status_container.write("---")
            
        status_container.update(label="Hoàn thành!", state="complete", expanded=False)
        answer_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        print("\n" + "="*30 + " LỊCH SỬ CHAT " + "="*30)
        for msg in st.session_state.messages:
            role = "NGƯỜI DÙNG" if msg["role"] == "user" else "CHATBOT"
            print(f"[{role}]: {msg['content']}")
        print("="*73 + "\n")

st.title("Advanced AI Agent (SQL + RAG + Python)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("VD: Doanh thu tháng này? Quy định nghỉ phép? Vẽ biểu đồ giá..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    asyncio.run(run_chat_logic(prompt))