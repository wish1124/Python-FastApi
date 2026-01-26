# graph.py
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from usage_tool import usage_tool
from search_tool_nltojson import extract_notice_query

# 환경 변수 로드
load_dotenv()

# =================================================================
# 1. State Definition
# =================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# =================================================================
# 2. Tool Definitions
# =================================================================
# search_tool is imported from search_tool_nltojson.py as extract_notice_query

tools = [extract_notice_query, usage_tool]

# =================================================================
# 3. Graph Nodes Setup
# =================================================================
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """LLM이 다음 행동(답변 or 툴호출)을 결정"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

def human_input_node(state: AgentState):
    """
    [시작 노드] 사용자 입력 확인용 (로그 출력)
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, HumanMessage):
        print(f"\n[System] 사용자 입력 수신: {last_msg.content[:20]}...")
    return None

# =================================================================
# 4. Graph Construction
# =================================================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("human_input", human_input_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 엣지 연결
workflow.add_edge(START, "human_input")
workflow.add_edge("human_input", "agent")

workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()

# 외부에서 import해서 쓸 수 있는 최종 앱 객체
graph_app = workflow.compile(checkpointer=memory)
