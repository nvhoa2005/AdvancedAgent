from typing import Annotated, TypedDict, Sequence
from pydantic import BaseModel, Field
from datetime import datetime
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from src.tools import (
    query_sql_db,
    search_policy_docs,
    python_chart_maker,
    get_db_schema,
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_out_of_scope: bool
    retry_count: int


class RouteResponse(BaseModel):
    reasoning: str = Field(
        description="Phân tích ngắn gọn tại sao câu hỏi thuộc hoặc không thuộc phạm vi."
    )
    is_out_of_scope: bool = Field(
        description="Kết quả cuối cùng: True nếu ngoài phạm vi, False nếu liên quan đến dữ liệu công ty."
    )


tools = [query_sql_db, search_policy_docs, python_chart_maker]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_with_tools = llm.bind_tools(tools)
llm_writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)

memory = MemorySaver()


def get_system_message():
    schema_info = get_db_schema()

    prompt = f"""Bạn là bộ não trung tâm của hệ thống Insight Agent.
    
    Nhiệm vụ: Phân tích yêu cầu của người dùng và SỬ DỤNG TOOL để lấy thông tin (nếu cần thiết).
    1. Nếu liên quan đến dữ liệu (SQL), chính sách (RAG), biểu đồ -> Gọi TOOL phù hợp.
    2. Nếu là câu hỏi chung (ví dụ: 'thời tiết', 'nấu ăn', 'tâm sự') -> KHÔNG gọi tool, hãy trả lời: 'GENERAL_CHAT'
    
    QUY TRÌNH SUY LUẬN ĐA BƯỚC:
    1. Phân tích yêu cầu -> Chọn Tool.
    2. Nếu kết quả Tool trả về là LỖI (Error), bạn phải:
    - Đọc kỹ thông báo lỗi.
    - Suy luận tại sao lỗi (ví dụ: nhầm tên cột, thiếu điều kiện JOIN).
    - Tự sửa lại câu lệnh và gọi lại Tool đó một lần nữa.
    3. Bạn có tối đa 3 lần thử lại cho mỗi yêu cầu.
    
    Các Tool có sẵn:
    1. query_sql_db: Lấy số liệu từ DB. Schema: {schema_info}.
    - LƯU Ý: Tuyệt đối KHÔNG dùng dấu chấm phẩy (;) cuối câu lệnh SQL.
    2. search_policy_docs: Tra cứu chính sách.
    3. python_chart_maker: Vẽ biểu đồ.
    
    HƯỚNG DẪN QUAN TRỌNG:
    - Nếu cần thông tin -> Gọi Tool.
    - Khi đã có kết quả từ Tool -> Hãy trả về thông tin dưới dạng **GẠCH ĐẦU DÒNG (Bullet points)** thô.
    - CHỈ TRẢ VỀ DỮ LIỆU. KHÔNG viết lời chào, KHÔNG viết câu dẫn, KHÔNG viết kết luận.
    - Ví dụ output mong muốn:
      * Doanh thu: 1000 USD
      * Số đơn: 50
    """
    return SystemMessage(content=prompt)


def agent_router_node(state: AgentState):
    """Node này thực hiện phân loại và lưu kết quả vào State"""
    messages = state["messages"]
    context = "\n".join([msg.content for msg in messages])
    last_user_message = messages[-1].content
    today = datetime.now().strftime("%d/%m/%Y")

    structured_llm = llm.with_structured_output(RouteResponse)

    check_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""Bạn là chuyên gia phân loại yêu cầu cho hệ thống AI Doanh nghiệp. 
        HÔM NAY LÀ: {today}.

        NHIỆM VỤ:
        Dựa vào ngữ cảnh và ĐẶC BIỆT chú ý đến câu hỏi cuối cùng của người dùng để quyết định xem nó có thể được giải quyết bằng các công cụ dữ liệu nội bộ hay không.

        DANH MỤC TRONG PHẠM VI (is_out_of_scope = False):
        - Mọi câu hỏi về doanh thu, đơn hàng, khách hàng, tồn kho, sản phẩm (Sử dụng SQL).
        - Mọi yêu cầu về quy định, chính sách công ty, phúc lợi, lương thưởng (Sử dụng RAG).
        - Yêu cầu vẽ biểu đồ, tính toán tỷ lệ tăng trưởng (Sử dụng Python).
        - LƯU Ý: Vì hiện nay là 2026, nên các câu hỏi về năm 2025 là DỮ LIỆU QUÁ KHỨ, hoàn toàn nằm trong phạm vi.

        DANH MỤC NGOÀI PHẠM VI (is_out_of_scope = True):
        - Chào hỏi xã giao (Hi, Hello), khen ngợi/chê bai không liên quan công việc.
        - Kiến thức thế giới chung (Thời tiết, nấu ăn, bóng đá, tin tức showbiz).
        - Câu hỏi về các công ty công nghệ khác (OpenAI, Google) trừ khi hỏi về sự tương tác với dữ liệu nội bộ.

        BẮT BUỘC: Nếu câu hỏi có chứa từ khóa liên quan đến 'doanh thu', 'bán hàng', 'quy định', 'bao nhiêu' -> Phải trả về is_out_of_scope = False.
        """,
            ),
            ("human", "{context}"),
        ]
    )

    final_prompt = check_prompt.format(context=context)
    result = structured_llm.invoke(final_prompt)

    print("\n--- [ROUTER LOG] ---")
    print(f"Câu hỏi: {last_user_message}")
    print(f"Suy luận: {result.reasoning}")
    print(f"Phân loại: {'Ngoài lề' if result.is_out_of_scope else 'Trong phạm vi'}")
    print("---------------------\n")

    return {
        "is_out_of_scope": result.is_out_of_scope, 
        "reasoning": result.reasoning
    }


def route_after_classification(state: AgentState):
    """Hàm này đóng vai trò làm ngã rẽ dựa trên State đã có"""
    if state.get("is_out_of_scope"):
        return "general_chat"
    return "agent"


def agent_node(state: AgentState):
    """Hàm này gọi tool và trả về dữ liệu thô"""
    messages = state["messages"]
    if "retry_count" not in state:
        state["retry_count"] = 0
        
    if not isinstance(messages[0], SystemMessage):
        sys_msg = get_system_message()
        messages = [sys_msg] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "retry_count": state["retry_count"] + 1}


def general_chat_node(state: AgentState):
    """Xử lý các câu hỏi ngoài phạm vi hệ thống"""

    messages = state["messages"]

    general_prompt = SystemMessage(
        content="""
    Bạn là một trợ lý ảo thông minh và vui vẻ. 
    Người dùng đang hỏi một câu hỏi ngoài phạm vi dữ liệu của công ty.
    Còn đây là các Tool mà bạn có thể trả lời nếu người dùng hỏi bạn có thể giúp được gì:
    1. query_sql_db: Lấy số liệu từ DB.
    2. search_policy_docs: Tra cứu chính sách.
    3. python_chart_maker: Vẽ biểu đồ.
    Hãy trả lời họ một cách tự nhiên, hữu ích dựa trên kiến thức chung của bạn.
    """
    )

    response = llm.invoke([general_prompt] + messages)
    return {"messages": [response]}


def final_answer_node(state: AgentState):
    """Viết câu trả lời cuối cùng"""
    messages = state["messages"]

    final_system_prompt = SystemMessage(
        content="""
    Bạn là Chuyên viên Chăm sóc Khách hàng chuyên nghiệp.
    Nhiệm vụ: Dựa trên các dữ liệu thô (gạch đầu dòng) mà hệ thống cung cấp ở trên, hãy VIẾT LẠI thành một câu trả lời hoàn chỉnh, tự nhiên cho người dùng.
    
    YÊU CẦU:
    1. KHÔNG được lặp lại nguyên văn các gạch đầu dòng. Hãy diễn giải thành lời văn.
    2. Nếu dữ liệu là con số, hãy làm tròn hoặc định dạng cho dễ đọc (ví dụ: 122,873.49 -> 122,873 USD).
    3. Văn phong lịch sự, thân thiện.
    4. Chỉ trả lời đúng trọng tâm câu hỏi.
    """
    )

    response = llm_writer.invoke([final_system_prompt] + messages)
    return {"messages": [response]}


def node_router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    
    if isinstance(last_message, ToolMessage) and ("Error" in last_message.content):
        if state.get("retry_count", 0) < 3:
            return "agent"
    
    return "final_answer"


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent_router", agent_router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("final_answer", final_answer_node)
workflow.add_node("general_chat", general_chat_node)

workflow.add_edge(START, "agent_router")

workflow.add_conditional_edges(
    "agent_router",
    route_after_classification,
    {"general_chat": "general_chat", "agent": "agent"},
)

workflow.add_conditional_edges(
    "agent", 
    node_router, 
    {"tools": "tools", "final_answer": "final_answer"}
)

workflow.add_edge("tools", "agent")
workflow.add_edge("general_chat", END)
workflow.add_edge("final_answer", END)

app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    try:
        os.makedirs("image", exist_ok=True)
        graph_image = app.get_graph().draw_mermaid_png()
        with open("image/agent_architecture.png", "wb") as f:
            f.write(graph_image)
        print("Đã lưu ảnh graph.")
    except Exception:
        pass
