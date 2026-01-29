# graph.py
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from usage_tool import usage_tool
from search_tool_nltojson import extract_notice_query

import json

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

사용자 질문을 다음 세 가지 유형 중 하나로 분류하라.

[유형 1: 사이트 기능 / 사용법 질문]
- 사이트의 화면, 메뉴, 버튼, 기능 사용 방법을 묻는 질문
- 이 경우 반드시 usage_tool을 호출하라.
- usage_tool의 반환값을 그대로 사용자에게 자연어로 전달하라.

예시:
- 로그인은 어디서 하나요?
- 공고 검색은 어떻게 하나요?
- PDF 다운로드 버튼이 안 보여요

[유형 2: 공고 조회 / 조건 추출 질문]
- 공고 검색, 지역, 금액, 기간 등 데이터 조회를 위한 조건을 묻는 질문
- 이 경우 반드시 extract_notice_query 도구를 호출하라.
- 도구가 반환한 JSON을 가공하거나 자연어로 설명하지 말고 그대로 출력하라.

예시:
- 부산에서 이번달 공고 보여줘
- 5억 이상 공고 낙찰하한율 평균 알려줘

[유형 3: 범위 밖 질문]

이 경우 너는 절대 추가 설명을 하지 않는다.
절대 추천, 정보 제공, 잡담을 하지 않는다.
절대 두 문장 이상 출력하지 않는다.

무조건 아래 문장만 출력하고 즉시 종료하라:

"저는 공공입찰 공고 조회와 사이트 이용 안내만 지원합니다."

중요 규칙:
- 세가지 유형 중 하나만 선택하라.
- 유형 1,2만 도구를 호출한다.
- 유형 3은 도구 호출을 금지한다.
- 두 개의 도구를 동시에 호출하지 마라.
"""
)

llm_postprocess=ChatOpenAI(model="gpt-5-nano",temperature=1)

SYSTEM_PROMPT_postprocess=SystemMessage(
    content="""
    너는 공공입찰 플랫폼 챗봇이다.

    지금부터 너에게 전달되는 입력은
    사용자의 질문이 아니라,
    백엔드에서 조회한 공고 검색 결과(JSON 데이터)이다.

    너의 임무는 다음과 같다:

    1. JSON 데이터를 읽고 핵심 정보를 자연어로 요약한다.
    2. 사용자가 이해할 수 있도록 공고 내용을 설명한다.
    3. 만약 결과가 비어 있다면 "조건에 맞는 공고가 없습니다"라고 답한다.
    4. JSON을 그대로 출력하지 말고 자연어로만 답한다.

    출력 예시:

    - "총 3건의 공고가 검색되었습니다."
    - "기초금액 평균은 약 3억 원입니다."
    - "낙찰하한율 최댓값은 87.5%입니다."

"""
)

def agent_node(state: AgentState):
    """LLM이 다음 행동(답변 or 툴호출)을 결정"""   
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def postprocess_node(state: AgentState):
    """공고 정보 조회 데이터를 llm을 통해 자연어로 풀어줌"""
    messages=[SYSTEM_PROMPT_postprocess]+state["messages"]
    reponse=llm_postprocess.invoke(messages)
    return {"messages":[reponse]}

def human_input_node(state: AgentState):
    """
    [시작 노드] 사용자 입력 확인용 (로그 출력)
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, HumanMessage):
        print(f"\n[System] 사용자 입력 수신: {last_msg.content[:20]}...")
    return None

def router_node(state: AgentState):
    """입력이 질문인지 db조회 결과인지 확인해주는 노드"""
    data = state["messages"][-1].content.strip()

    try:
        data = json.loads(data)

        if data.get("type") in ["notice_result", "report"]:
            return "postprocess"

    except:
        pass

    return "agent"


# =================================================================
# 4. Graph Construction
# =================================================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("human_input", human_input_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("postprocess", postprocess_node)

# 엣지 연결
workflow.add_edge(START, "human_input")

workflow.add_conditional_edges("human_input",router_node,
    {
        "agent": "agent",
        "postprocess": "postprocess"
    })

workflow.add_conditional_edges("agent", tools_condition,{"tools":"tools","__end__":END})

workflow.add_edge("postprocess",END)
workflow.add_edge("tools", END)

memory = MemorySaver()

# 외부에서 import해서 쓸 수 있는 최종 앱 객체
graph_app = workflow.compile(checkpointer=memory)
