import operator
import os
import getpass
from typing import Annotated, List, TypedDict, Union

# .env 파일 로드를 위한 라이브러리 (pip install python-dotenv 필요)
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# =================================================================
# 0. API Key Configuration
# =================================================================
def setup_api_keys():
    """API 키를 설정하는 함수입니다."""
    # 1. .env 파일이 있다면 로드
    load_dotenv()
    
    # 2. OpenAI API Key 확인 및 입력
    if not os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API Key가 환경 변수에 없습니다.")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")
    
    # 3. (옵션) 공공데이터포털 등 다른 API 키가 필요한 경우 추가
    if not os.environ.get("DATA_PORTAL_API_KEY"):
        # os.environ["DATA_PORTAL_API_KEY"] = getpass.getpass("공공데이터 API Key를 입력하세요: ")
        pass # 현재는 주석 처리

# 실행 시 키 설정 먼저 수행
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
    """
    각 지역별 공고를 알려주는 노드입니다. 이 노드는 openapi로부터 값을 조회해 최근 순으로 3건 알려줍니다.
    """
    # 실제 구현 시: service_key = os.environ["DATA_PORTAL_API_KEY"]
    return f"[Mock] '{query}'에 대한 지역별 공고 검색 결과: [서울 공고1, 부산 공고2, ...]"

@tool
def search_by_budget_tool(query: str):
    """
    예산 조건 기준으로 공고를 검색하는 툴 노드입니다.
    """
    return f"[Mock] '{query}'에 대한 예산 기준 검색 결과: [1억 이하 공고 리스트...]"

@tool
def closing_bids_tool(data_query: str):
    """
    마감이 임박한 공고들을 알려주는 노드입니다.
    """
    return "[Mock] 마감 임박 공고: [D-1 공고 A, D-Day 공고 B...]"

@tool
def usage_tool(param: str):
    """
    api 정의서 및 사전에 입력된 웹페이지 캡쳐 화면을 통해 웹 기능들을 설명해줍니다.
    """
    return f"[Mock] 사용법 안내 ({param}): '이 기능은 ~게 사용합니다.'"

# 툴 리스트 정의
tools = [regional_bids_tool, search_by_budget_tool, closing_bids_tool, usage_tool]

# =================================================================
# 3. LLM & Graph Nodes Setup
# =================================================================

# LLM 초기화 (gpt-5-nano는 아직 없으므로 gpt-4o로 설정, 실제 사용 가능한 모델명으로 변경하세요)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

# =================================================================
# 4. Graph Construction
# =================================================================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# =================================================================
# 5. Execution Example
# =================================================================
if __name__ == "__main__":
    
    # 그래프 시각화 저장
    try:
        graph_image = app.get_graph().draw_mermaid_png()
        with open("graph_visualization.png", "wb") as f:
            f.write(graph_image)
        print("[System] 'graph_visualization.png' 저장 완료")
    except Exception as e:
        print(f"[System] 그래프 이미지 저장 실패 (무시 가능): {e}")

    # 대화 실행 테스트
    print("\n--- 에이전트 실행 시작 ---")
    inputs = {"messages": [HumanMessage(content="서울 지역의 마감 임박 공고 3개만 찾아줘.")]}
    config = {"configurable": {"thread_id": "thread-1"}}

    for event in app.stream(inputs, config=config):
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            # 결과 메시지 내용만 깔끔하게 출력 (옵션)
            # if "messages" in value:
            #     print(value["messages"][-1].content)
