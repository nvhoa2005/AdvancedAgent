from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage

from core.state import AgentState
from agent.nodes import AgentNodes

class InsightAgentWorkflow:
    """Class quản lý việc xây dựng và biên dịch LangGraph."""
    
    def __init__(self, nodes: AgentNodes, tools: list):
        self.nodes = nodes
        self.tool_node = ToolNode(tools)
        self.memory = MemorySaver()
        self.workflow = StateGraph(AgentState)
        self._build_graph()
    
    @staticmethod
    def _route_after_input_guard(state: AgentState):
        if state.get("is_out_of_scope") is True:
            return "general_chat"
        return "agent_router"

    @staticmethod
    def _route_after_classification(state: AgentState):
        if state.get("is_out_of_scope"):
            return "general_chat"
        return "query_transform"

    @staticmethod
    def _node_router(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "tools"
        
        if isinstance(last_message, ToolMessage) and ("Error" in last_message.content):
            if state.get("retry_count", 0) < 3:
                return "agent"
        
        return "final_answer"

    
    def _build_graph(self):
        self.workflow.add_node("input_guardrail", self.nodes.input_guardrail)
        self.workflow.add_node("agent_router", self.nodes.agent_router)
        self.workflow.add_node("query_transform", self.nodes.query_transform)
        self.workflow.add_node("agent", self.nodes.agent)
        self.workflow.add_node("tools", self.tool_node)
        self.workflow.add_node("final_answer", self.nodes.final_answer)
        self.workflow.add_node("general_chat", self.nodes.general_chat)
        self.workflow.add_node("output_guardrail", self.nodes.output_guardrail)

        self.workflow.add_edge(START, "input_guardrail")
        
        self.workflow.add_conditional_edges(
            "input_guardrail",
            self._route_after_input_guard,
            {"general_chat": "general_chat", "agent_router": "agent_router"}
        )

        self.workflow.add_conditional_edges(
            "agent_router",
            self._route_after_classification,
            {"general_chat": "general_chat", "query_transform": "query_transform"}
        )

        self.workflow.add_edge("query_transform", "agent")

        self.workflow.add_conditional_edges(
            "agent", 
            self._node_router, 
            {"tools": "tools", "final_answer": "final_answer"}
        )

        self.workflow.add_edge("tools", "agent")
        self.workflow.add_edge("final_answer", "output_guardrail")
        self.workflow.add_edge("general_chat", "output_guardrail")
        
        self.workflow.add_edge("output_guardrail", END)

    def compile(self):
        """Biên dịch và trả về Graph App để sử dụng."""
        return self.workflow.compile(checkpointer=self.memory)

    def save_graph_image(self, path="static/agent_architecture.png"):
        """Hàm hỗ trợ lưu ảnh sơ đồ kiến trúc."""
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            app = self.compile()
            graph_image = app.get_graph().draw_mermaid_png()
            with open(path, "wb") as f:
                f.write(graph_image)
            print(f"Đã lưu ảnh graph tại {path}")
        except Exception as e:
            print(f"Không thể lưu ảnh graph: {e}")