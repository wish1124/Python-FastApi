from langchain.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-nano", temperature=1)

@tool
def extract_notice_query(user_query: str) -> dict:
    """
    이 도구는 사용자의 질문을 공고 조회를 위한 필터 조건 JSON 객체로 변환한다.
    공고 조회가 아닌 질문에는 사용하지 않는다.
    """
    prompt="""
    사용자의 질문을 공고 조회를 위한 필터 조건 json 객체로 변환한다.
    반드시 아래 규칙에 따라 JSON 객체 하나만 생성해야 한다.

    이 도구는 “조건을 선언”만 하며,
    실제 날짜 계산, 달력 처리 등은 서버에서 수행한다.

    중요 규칙:
    - 출력 JSON의 필드명은 반드시 아래에 정의된 이름만 사용해야 한다.
    - 정의되지 않은 필드명을 새로 만들면 안 된다.
    - 명시하지 않는 필드는 반드시 null로 설정한다.
    - 출력은 반드시 JSON 객체 하나만 반환한다. 설명 문장, 주석, 자연어는 절대 포함하지 않는다.
    - 시간 관련 조건은 반드시 timeRange로만 표현한다.
    - 필드의 의미를 추론해 임의로 확장하거나 축약하지 않는다.
    - 숫자형 날짜는 yyyyMMddHHmm 형식만 사용한다. (예: 202601231359)
    - LLM은 날짜 계산을 하지 않고 반드시 선언형 구조로만 표현한다.
    - 사용자의 질문에서 공고 조회 조건을 합리적으로 추출할 수 없는 경우 ex)조건 필드는 있으나 op / 방향 / 기준(base)이 모호하거나 숫자 범위가 상식적으로 충돌하는 경우, 사용자의 질문이 공고 조회 조건으로 해석 불가능한 경우 모든 필드를 null로 설정하고 intent는 list로 설정한다.

    이 도구의 출력은 백엔드 서버에서 직접 실행된다.
    출력 JSON이 잘못되면 잘못된 DB 조회로 이어진다.
    --------------------------------------------------
    timeRange (시간 조건 선언)

    timeRange는 “언제의 공고를 조회할 것인가”를
    계산 없이 선언적으로 표현하는 필드이다.

    timeRange는 하나만 설정할 수 있다.

    기준 필드(base):
    - startDate : 공고 시작일 기준 조회
    - endDate   : 공고 마감일 기준 조회
    - openDate  : 공고 개찰일 기준 조회

    base 판단 규칙:

    1. 사용자가 명시적으로 언급한 날짜 기준이 있으면 그것을 최우선으로 사용한다.
      - "시작", "시작일" → startDate
      - "마감", "마감일", "마감 기준" → endDate
      - "개찰", "개찰일" → openDate

    2. "까지", "부터", "이전", "이후", "이내" 같은 표현은
      기간의 방향·범위를 나타낼 뿐 base를 결정하지 않는다.

    3. 명시적인 기준이 없는 경우 (예: “언제부터 언제까지 공고 보여줘”) base는 openDate로 설정한다.

    4. 둘 이상의 기준이 동시에 언급된 경우, 문장에서 직접 수식되는 기준을 우선한다.

    - “~부터” → from 사용
    - “~까지” → to 사용
    - from과 to는 각각 독립적으로 설정한다.

    - from 또는 to 중 하나만 존재해도 된다.

    - from과 to는 서로 다른 kind를 가질 수 있다.

    - 단, from 내부 / to 내부에서는 kind를 혼용할 수 없다.



    -from 또는 to는 숫자 날짜 또는 calendar 선언 객체를 사용할 수 있다.

    "kind": "absolute | calendar",

    kind별 의미:
      -absolute:절대 시점을 의미한다. value에 yyyyMMddHHmm 형식의 숫자를 사용한다.
      {
        "kind": "absolute",
        "value": 202601011230
      }
      -calendar: 상대적인 달력 기준 시점을 의미한다. unit / offset / position을 사용한다.
      -unit: 기준 단위 (day, week, month, year)
      -offset: 기준 시점으로부터의 이동 값
      -position: 해당 단위의 시작(start) 또는 끝 (end)

    - kind가 absolute인 경우 value만 사용하며,
      unit / offset / position은 반드시 null이다.

    - kind가 calendar인 경우 unit / offset / position만 사용하며,
      value는 반드시 null이다.

    calendar kind의 기준 시점(now)은
    필터를 해석하는 서버 시점이며,
    LLM은 기준 시점을 계산하거나 추론하지 않는다.

    의미 예시:
    - "이번 주 말까지" → to: { kind: calendar, unit: week, offset: 0, position: end }
    - "다음 달 초부터" → from: { kind: calendar, unit: month, offset: 1, position: start }
    - "최근 7일 이내" → from: { kind: calendar, unit: day, offset: -7, position: null }
    값을 해석하거나 날짜로 변환하지 않는다.

    --------------------------------------------------

    연산자(op) 의미
    - "gte": 이상 (>=)
    - "gt": 초과 (>)
    - "lte": 이하 (<=)
    - "lt": 미만 (<)
    - "eq": 동일 (=)

    --------------------------------------------------

    금액 단위(억, 만 등)가 포함된 경우
    LLM은 이를 원 단위 숫자로 변환하여 value에 선언한다.
    -“기초금액 10억 이상” → 1000000000

    비율 관련 필드(minimumBidRate)는
    항상 0과 100 사이의 실수 값으로 표현한다.

    - 사용자가 "%", "퍼센트", "이상", "이하"와 함께
      1 이상 숫자를 언급한 경우:
      → 해당 값은 이미 퍼센트로 표현된 값으로 간주하고 그대로 사용한다.

      예:
      "낙찰하한율 88.5 이상" → 88.5
      "낙찰하한율 90 이하" → 90

    - 사용자가 1 이하의 소수를 직접 언급한 경우:
      → 비율로 표현된 값으로 간주하고 100을 곱한다.

      예:
      "낙찰하한율 0.12 미만" → 12

    수치 관련 필드에서
    "이상", "초과" → from 사용
    "이하", "미만" → to 사용

    from.op 은 반드시 "gte" 또는 "gt" 만 허용한다.
    단, 단일 값 비교의 경우(from만 사용하는 경우)에 한해
    from.op = eq 를 예외적으로 허용한다.
    to.op   는 반드시 "lte" 또는 "lt" 만 허용한다.

    "A 이상 B 이하"와 같이 범위가 명시된 경우:
    - 반드시 from과 to를 모두 설정한다.
    - 하나의 조건으로 축약하지 않는다.

    "A" 하나만 명시된 경우:
    - eq가 아닌 경우, 방향에 따라 from 또는 to만 설정한다.

    "~같", "~인", "~인 공고" 등
    단일 값 비교의 경우:

    - from과 to를 모두 사용하지 않고
    - from.op = eq 만 사용하며 from.to 는 반드시 null 이다.

    --------------------------------------------------
    intent는 사용자가 “무엇을 원하는가”를 선언한다.
    서버는 intent를 해석하지 않고 그대로 실행 분기만 한다.

    limit은 LIST intent에서만 의미가 있다.
    "단일 대상 조회”라고 판단한 경우 limit=1을 선언한다.
    intent가 count 또는 aggregate인 경우 limit은 반드시 null로 설정한다.

    aggregate는 count / aggregate intent에서만 사용한다.
    list와 동시에 사용하지 않는다.




    사용 가능한 필드:
    {
  "intent": "list | count | aggregate",
  "limit": number | null,
  "filter": {
    "bidRealId": string | null,
    "region": string | null,
    "organization": string | null,

    "estimatePrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "basicPrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "minimumBidRate": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "bidRange": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "timeRange": {
      "base": "startDate | endDate | openDate",
      "from": {
        "kind": "absolute | calendar",
        "value": number | null,
        "unit": "day | week | month | year" | null,
        "offset": number | null,
        "position": "start | end" | null
      },
      "to": {
        "kind": "absolute | calendar",
        "value": number | null,
        "unit": "day | week | month | year" | null,
        "offset": number | null,
        "position": "start | end" | null
      }
    } | null
  },
  "aggregate": {
    "type": "count | avg | sum | min | max",
    "field": "estimatePrice | basicPrice | minimumBidRate | bidRange"
  } | null
}

    구조화 결과 예시:

    예시 1)
    입력: "공고번호 20240123456-000 내용 알려줘"

    출력:
    {
      "intent": "list",
      "limit": 1,
      "filter": {
        "bidRealId": "20240123456-000",
        "region": null,
        "organization": null,
        "estimatePrice": null,
        "basicPrice": null,
        "minimumBidRate": null,
        "bidRange": null,
        "timeRange": null
      },
      "aggregate": null
    }

    예시 2)
    입력: "부산 지역에 2026년 1월 1일부터 다음 달 말까지 개찰일 기준 공고의 개수를 보여줘"

    출력:
    {
      "intent": "count",
      "limit": null,
      "filter": {
        "bidRealId": null,
        "region": "부산",
        "organization": null,
        "estimatePrice": null,
        "basicPrice": null,
        "minimumBidRate": null,
        "bidRange": null,
        "timeRange": {
          "base": "openDate",
          "from": {
            "kind": "absolute",
            "value": 202601010000,
            "unit": null,
            "offset": null,
            "position": null
          },
          "to": {
            "kind": "calendar",
            "value": null,
            "unit": "month",
            "offset": 1,
            "position": "end"
              }
            }
      },
      "aggregate": {
        "type": "count",
        "field": null
      }
    }


    예시 3)
    입력: "마감일이 202601010000부터 202601011234인 서울에서 기초금액 10억 이상이고 낙찰하한율이 0.8 이상이고 95%보다 작은 공고의 기초금액 평균을 알려줘"

    출력:
    {
      "intent": "aggregate",
      "limit": null,
      "filter": {
        "bidRealId": null,
        "region": "서울",
        "organization": null,
        "estimatePrice": null,
        "basicPrice": {
          "from":{"op": "gte", "value": 1000000000},
          "to": null
        },
        "minimumBidRate":{
            "from":{"op": "gte", "value": 80},
            "to":{"op": "lt", "value":95}
          },
        "bidRange": null,
        "timeRange": {
        "base": "endDate",
        "from": {
          "kind": "absolute",
          "value": 202601010000,
          "unit": null,
          "offset": null,
          "position": null
          },
        "to": {
          "kind": "absolute",
          "value": 202601011234,
          "unit": null,
          "offset": null,
          "position": null
        }
        }
        },
        "aggregate": {
          "type" : "avg",
          "field" : "basicPrice"
        }
    }
    """ 
    prompt+=f"""
    질문:
    {user_query}
    
    출력은 json 객체 하나만 반환한다.
    """

    response = llm.invoke(prompt)
    return response.content