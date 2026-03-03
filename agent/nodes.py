from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import AgentState, GuardrailResponse, RouteResponse
from core.prompts import (
    SYSTEM_PROMPT_TEMPLATE, 
    INPUT_GUARDRAIL_PROMPT, 
    OUTPUT_GUARDRAIL_PROMPT,
    QUERY_TRANSFORM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    GENERAL_CHAT_PROMPT,
    FINAL_ANSWER_PROMPT
)

class AgentNodes:
    """Class chứa logic thực thi của từng node trong LangGraph."""
    def __init__(self, llm, llm_writer, tools, db_schema: str):
        self.llm = llm
        self.llm_writer = llm_writer
        self.tools = tools
        self.db_schema = db_schema
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _get_system_message(self):
        """Khởi tạo System Prompt với schema của DB."""
        prompt = SYSTEM_PROMPT_TEMPLATE.format(schema_info=self.db_schema)
        return SystemMessage(content=prompt)

    def input_guardrail(self, state: AgentState):
        last_user_message = state["messages"][-1].content
        structured_llm = self.llm.with_structured_output(GuardrailResponse)
        
        prompt = INPUT_GUARDRAIL_PROMPT.format(last_user_message=last_user_message)
        check = structured_llm.invoke(prompt)
        
        return {
            "is_out_of_scope": not check.is_safe,
            "reasoning": check.reasoning,
            "is_safe": check.is_safe 
        }

    def output_guardrail(self, state: AgentState):
        last_ai_message = state["messages"][-1].content
        prompt = OUTPUT_GUARDRAIL_PROMPT.format(last_ai_message=last_ai_message)
        
        response = self.llm_writer.invoke(prompt)
        return {"messages": [response]}

    def query_transform(self, state: AgentState):
        messages = state["messages"]
        last_user_message = messages[-1].content
        today = datetime.now().strftime("%d/%m/%Y")
        context = " ".join([msg.content for msg in messages[:-1]])
        
        prompt = QUERY_TRANSFORM_PROMPT.format(
            today=today,
            last_user_message=last_user_message,
            context=context
        )
        
        transformed = self.llm.invoke(prompt)
        print("\n--- [QUERY TRANSFORM] ---")
        print(f"Gốc: {last_user_message}")
        print(f"Mới: {transformed.content}")
        print("--------------------------\n")
        
        return {"transformed_query": transformed.content}

    def agent_router(self, state: AgentState):
        messages = state["messages"]
        context = "\n".join([msg.content for msg in messages])
        last_user_message = messages[-1].content
        today = datetime.now().strftime("%d/%m/%Y")

        structured_llm = self.llm.with_structured_output(RouteResponse)
        prompt = ROUTER_SYSTEM_PROMPT.format(today=today)
        
        messages_to_invoke = [
            SystemMessage(content=prompt),
            HumanMessage(content=context)
        ]
        
        result = structured_llm.invoke(messages_to_invoke)

        print("\n--- [ROUTER LOG] ---")
        print(f"Câu hỏi: {last_user_message}")
        print(f"Suy luận: {result.reasoning}")
        print(f"Phân loại: {'Ngoài lề' if result.is_out_of_scope else 'Trong phạm vi'}")
        print("---------------------\n")

        return {
            "is_out_of_scope": result.is_out_of_scope, 
            "reasoning": result.reasoning
        }

    def agent(self, state: AgentState):
        messages = state["messages"]
        
        if state.get("transformed_query"):
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    messages[i] = HumanMessage(content=state["transformed_query"])
                    break
        
        if "retry_count" not in state:
            state["retry_count"] = 0
            
        if not isinstance(messages[0], SystemMessage):
            sys_msg = self._get_system_message()
            messages = [sys_msg] + messages

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "retry_count": state["retry_count"] + 1}

    def general_chat(self, state: AgentState):
        messages = state["messages"]
        reasoning = state.get("reasoning", "")

        prompt = GENERAL_CHAT_PROMPT.format(reasoning=reasoning)
        general_prompt = SystemMessage(content=prompt)

        response = self.llm.invoke([general_prompt] + messages)
        return {"messages": [response]}

    def final_answer(self, state: AgentState):
        messages = state["messages"]
        final_system_prompt = SystemMessage(content=FINAL_ANSWER_PROMPT)

        response = self.llm_writer.invoke([final_system_prompt] + messages)
        return {"messages": [response]}