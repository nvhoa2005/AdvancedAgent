from typing import Annotated, TypedDict, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_out_of_scope: bool
    retry_count: int
    reasoning: str
    is_safe: bool
    transformed_query: str

class RouteResponse(BaseModel):
    reasoning: str = Field(
        description="Phân tích ngắn gọn tại sao câu hỏi thuộc hoặc không thuộc phạm vi."
    )
    is_out_of_scope: bool = Field(
        description="Kết quả cuối cùng: True nếu ngoài phạm vi, False nếu liên quan đến dữ liệu công ty."
    )
    
class GuardrailResponse(BaseModel):
    is_safe: bool = Field(description="True nếu yêu cầu/nội dung an toàn, False nếu vi phạm chính sách.")
    reasoning: str = Field(description="Lý do cụ thể nếu không an toàn (ví dụ: Prompt Injection, PII leakage).")
    action: str = Field(description="Hành động: 'proceed', 'refuse', hoặc 'mask_data'.")