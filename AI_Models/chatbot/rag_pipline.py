import operator
import os
import getpass
from typing import Annotated, List, TypedDict, Union

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# =================================================================
# 0. API Key Configuration
# =================================================================
def setup_api_keys():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API Key가 환경 변수에 없습니다.")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

setup_api_keys()

# =================================================================
# 1. State Definition
# =================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# =================================================================
# 2. Tool Definitions
# =================================================================
@tool
def regional_bids_tool(query: str):
    """각 지역별 공고를 알려주는 노드입니다. 최근 순으로 3건 알려줍니다."""
    return f"[Mock] '{query}' 지역별 공고 검색 결과: [서울 공고1, 부산 공고2]"

@tool
def search_by_budget_tool(query: str):
    """예산 조건 기준으로 공고를 검색하는 툴 노드입니다."""
    return f"[Mock] '{query}' 예산 기준 검색 결과: [1억 이하 공고 리스트]"

@tool
def closing_bids_tool(data_query: str):
    """마감이 임박한 공고들을 알려주는 노드입니다."""
    return "[Mock] 마감 임박 공고: [D-1 공고 A]"

@tool
def usage_tool(param: str):
    """기능 사용법을 설명해주는 도구입니다."""
    return f"[Mock] 사용법 안내 ({param})"

tools = [regional_bids_tool, search_by_budget_tool, closing_bids_tool, usage_tool]

# =================================================================
# 3. Graph Nodes
# =================================================================

llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """LLM이 다음 행동(답변 or 툴호출)을 결정"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

def human_input_node(state: AgentState):
    """
    [시작 노드]
    사용자 입력이 State에 주입된 후 가장 먼저 실행되는 노드입니다.
    특별한 로직 없이 메시지가 잘 들어왔는지 로그만 찍고 에이전트로 넘깁니다.
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, HumanMessage):
        print(f"\n[System] 사용자 입력 수신 확인: {last_msg.content[:20]}...")
    return None 

# =================================================================
# 4. Graph Construction
# =================================================================

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("human_input", human_input_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# [엣지 변경] 시작 -> 사람(입력 확인) -> 에이전트
workflow.add_edge(START, "human_input")
workflow.add_edge("human_input", "agent")

# 에이전트 -> (툴 or 종료)
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent") 

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# =================================================================
# 5. Execution Logic
# =================================================================
if __name__ == "__main__":
    print("--- LangGraph Chatbot 시작 ---")
    
    config = {"configurable": {"thread_id": "web-session-1"}}
    
    # [수정] 최초 실행 시 input()을 먼저 호출하여 사람이 입력하게 함
    user_input = input("\n[Start] 질문을 입력하세요 ('q' 종료): ")

    while True:
        if user_input.lower() in ["q", "quit", "exit"]:
            print("대화를 종료합니다.")
            break
            
        # 그래프 실행 (사용자 입력 주입 -> human_input -> agent ...)
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # 스트리밍 실행
        for event in app.stream(inputs, config=config):
            for key, value in event.items():
                # 노드 실행 로그 출력
                if key == "human_input":
                    pass # 이미 내부에서 출력함
                elif key == "agent":
                    msg = value["messages"][-1]
                    if msg.content:
                        print(f"\n[Agent]: {msg.content}")
                elif key == "tools":
                    print(f"\n[Tools]: 도구 실행 완료")

        # 다음 입력을 받음
        user_input = input("\n[User] >>> ")
