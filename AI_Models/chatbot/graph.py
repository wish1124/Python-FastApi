# graph.py
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
llm = ChatOpenAI(model="gpt-5-nano",temperature=1)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content="""
너는 공공입찰 플랫폼 챗봇의 중앙 판단 에이전트이다.

사용자 질문을 다음 두 가지 유형 중 하나로 분류하라.

[유형 1: 사이트 기능 / 사용법 질문]
- 사이트의 화면, 메뉴, 버튼, 기능 사용 방법을 묻는 질문
- 이 경우 반드시 usage_tool을 호출하라.
- usage_tool의 반환값을 그대로 사용자에게 자연어로 전달하라.

[유형 2: 공고 조회 / 조건 추출 질문]
- 공고 검색, 지역, 금액, 기간 등 데이터 조회를 위한 조건을 묻는 질문
- 이 경우 반드시 extract_notice_query 도구를 호출하라.
- 도구가 반환한 JSON을 가공하거나 자연어로 설명하지 말고 그대로 출력하라.

중요 규칙:
- 두 유형 중 하나만 선택하라.
- 두 개의 도구를 동시에 호출하지 마라.
"""
)

def agent_node(state: AgentState):
    """LLM이 다음 행동(답변 or 툴호출)을 결정"""
    last= state["messages"][-1]       

    #tool 결과로 extract_notice_query면 판단 중단
    if isinstance(last, ToolMessage):
        print("tool 호출됨:", last.name)
        #return {"messages": [last]}

    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

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
