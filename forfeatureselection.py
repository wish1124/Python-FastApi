import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

'''

-> "공고번호", "차수","공고명" : 공고 구분용
->"입찰준비기간"(공고시작~입찰시작) , "공고개찰기간"(공고시작~개찰기간) , "자격등록기간"(공고시작~자격등록마감) : 이 기간들이 짧을수록 까다로워서 참여율 감소

-> "기초금액", "예가범위" : 논문 구현용
-> "예산대비 추정가 비율"(추정가/예산): 발주처가 예산 대비 얼마나 타이트하게 가격을 잡았는지 1에 가까울 수록 여유가 없어 경쟁률 낮을  0에 여유가 크니 경쟁률 높을 것,
   "관급비 비중"(관급비/추정가): 값이 클 수록 낙찰률 들어갈 것,
   "순공사비 (=실질 시공금액,추정가-관급비-세금)":값이 클수록 덤핑할 확률 증가,
   "VAT 비율(세금/추정가)": 비율이 이상하면 시스템도 이상할 것
   "난이도계수": 공고 난이도를 나타내는 듯함
'''

# ==========================================
# 설정값
# ==========================================
SERVICE_KEY = "4244f2d0585c4637ce8b73310c8d6d4e70fb663f8c0c5d5ab31d6c43271e2d8a"
numOfRows = 500
inqryBgnDt = "202512010000"
inqryEndDt = "202512011159"
filename = "1_1_1.csv"

# 세션 생성 (연결 재사용으로 속도 향상)
session = requests.Session()

# ==========================================
# 1. 공사 조회
# ==========================================
url_time = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk"
all_items = []
page_no = 1

print("Dataset 1: 데이터 수집 시작...")
while True:
    params = {
        "serviceKey": SERVICE_KEY, "pageNo": page_no, "numOfRows": numOfRows,
        "inqryDiv": 1, "inqryBgnDt": inqryBgnDt, "inqryEndDt": inqryEndDt, "type": "json"
    }
    try:
        res = session.get(url_time, params=params).json()
        body = res.get("response", {}).get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict): items = items.get("item", [])
        if not isinstance(items, list): items = [items] if items else []

        if not items: break
        all_items.extend(items)

        total_count = body.get("totalCount", 0)
        print(f" - [1단계] {len(all_items)} / {total_count} 수집 중...")

        if len(all_items) >= total_count: break
        page_no += 1
    except Exception as e:
        print(f"Error in Step 1: {e}")
        break

df_time = pd.DataFrame(all_items)
if df_time.empty:
    print("데이터가 없어 종료합니다.")
    exit()

# 전처리
date_cols = [
    "bidNtceDt", "bidBeginDt", "bidClseDt",
    "opengDt", "bidQlfctRgstDt"
]

for col in date_cols:
    df_time[col] = pd.to_datetime(df_time[col], errors="coerce")

df_time["prep_days"] = (df_time["bidBeginDt"] - df_time["bidNtceDt"]).dt.total_seconds() / 86400
df_time["ntce_to_open_days"] = (df_time["opengDt"] - df_time["bidNtceDt"]).dt.total_seconds() / 86400
df_time["qlfct_margin_days"] = ( df_time["bidQlfctRgstDt"] - df_time["bidNtceDt"]).dt.total_seconds() / 86400

derived_cols = ["prep_days","ntce_to_open_days","qlfct_margin_days"]

for col in derived_cols:
    df_time[col] = df_time[col].clip(lower=0)
df_time[derived_cols] = df_time[derived_cols].fillna(df_time[derived_cols].median())


num_cols=[
    "govsplyAmt","presmptPrce","bdgtAmt"
]

for col in num_cols:
  df_time[col]=pd.to_numeric(df_time[col],errors="coerce")

df_time["gov_supply_ratio"] = (df_time["govsplyAmt"].astype(float) / df_time["presmptPrce"].astype(float))
df_time["budget_to_est_ratio"] = (df_time["presmptPrce"].astype(float) / df_time["bdgtAmt"].astype(float))
df_time["net_construction_cost"] = (df_time["presmptPrce"].astype(float)-df_time["govsplyAmt"].astype(float)-df_time["VAT"].astype(float))
df_time["vat_ratio"] = (df_time["VAT"].astype(float) / df_time["presmptPrce"].astype(float))






df_time = df_time.rename(columns={
    "bidNtceNo": "공고번호",
    "bidNtceOrd": "차수",
    "bidNtceNm": "공고명",
    "prep_days": "입찰준비기간",
    "ntce_to_open_days": "공고개찰기간",
    "qlfct_margin_days": "자격등록기간",
    "gov_supply_ratio":"관급비비중",
    "presmptPrce":"추정가격",
    "sucsfbidLwltRate": "낙찰하한율",
    "budget_to_est_ratio":"예산대비추정가",
    "net_construction_cost": "순공사비",
    "vat_ratio": "VAT비율",
    "bidPrtcptLmtYn":"참가제한",
    "indstrytyLmtYn":"업종제한",
    "cnstrtsiteRgnNm":"공사지역",
    "incntvRgnNm1":"가산점지역1",
    "incntvRgnNm2":"가산점지역2",
    "incntvRgnNm3":"가산점지역3",
    "incntvRgnNm4":"가산점지역4",
    "rgnDutyJntcontrctYn":"지역의무공동계약여부",
    "reNtceYn":"재공고여부"
})
df_time = df_time[["공고번호", "차수","공고명","추정가격","낙찰하한율" ,"예산대비추정가","관급비비중","순공사비","VAT비율","입찰준비기간", "공고개찰기간","자격등록기간","참가제한","업종제한","공사지역","가산점지역1","가산점지역2","가산점지역3","가산점지역4","지역의무공동계약여부","재공고여부"]]
print(f"✓ 1단계 완료: {len(df_time)}건")

# ==========================================
# 2. 공사 금액 조회
# ==========================================
url_bssis = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwkBsisAmount"
all_items = []
page_no = 1

print("Dataset 2: 각종 금액 수집 시작...")
while True:
    params = {
        "serviceKey": SERVICE_KEY, "pageNo": page_no, "numOfRows": numOfRows,
        "inqryDiv": 1, "inqryBgnDt": inqryBgnDt, "inqryEndDt": inqryEndDt, "type": "json"
    }
    try:
        res = session.get(url_bssis, params=params).json()
        body = res.get("response", {}).get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict): items = items.get("item", [])
        if not isinstance(items, list): items = [items] if items else []

        if not items: break
        all_items.extend(items)

        total_count = body.get("totalCount", 0)
        print(f" - [1단계] {len(all_items)} / {total_count} 수집 중...")

        if len(all_items) >= total_count: break
        page_no += 1
    except Exception as e:
        print(f"Error in Step 1: {e}")
        break

df_bssis = pd.DataFrame(all_items)
if df_bssis.empty:
    print("데이터가 없어 종료합니다.")
    exit()

# 전처리
df_bssis = df_bssis[["bidNtceNo","bidNtceOrd", "bidNtceNm", "bssamt", "rsrvtnPrceRngBgnRate", "rsrvtnPrceRngEndRate","dfcltydgrCfcnt","sftyMngcst","qltyMngcst"]]
df_bssis["예가범위"] = df_bssis[["rsrvtnPrceRngBgnRate", "rsrvtnPrceRngEndRate"]].apply(
    lambda x: max(abs(pd.to_numeric(x.iloc[0], errors="coerce")), abs(pd.to_numeric(x.iloc[1], errors="coerce"))), axis=1
)


df_bssis["안전관리비비율"] = (df_bssis["sftyMngcst"].astype(float) / df_bssis["bssamt"].astype(float))
df_bssis["품질관리비비율"] = (df_bssis["qltyMngcst"].astype(float) / df_bssis["bssamt"].astype(float))




df_bssis = df_bssis.rename(columns={
    "bidNtceNo": "공고번호",
    "bidNtceOrd": "차수",
    "bidNtceNm": "공고명",
    "bssamt": "기초금액",
    "dfcltydgrCfcnt": "난이도계수"
})
df_bssis = df_bssis[["공고번호", "차수","공고명", "기초금액", "예가범위","난이도계수","안전관리비비율","품질관리비비율"]]
print(f"✓ 2단계 완료: {len(df_bssis)}건")

# ==========================================
# 3. 병합
# ==========================================
df_all = df_bssis.merge(df_time, on=["공고번호", "차수"], how="left")

# ==========================================
# 4. 낙찰가 (기간 검색으로 변경 - API 절약 & 속도 UP)
# ==========================================
# 중요: 여기도 기존 코드는 반복문이었으나, '기간 검색'이 되므로 한 번에 가져옵니다.
url_scsbid = "https://apis.data.go.kr/1230000/as/ScsbidInfoService/getScsbidListSttusCnstwkPPSSrch"
scsbid_rows = []
page_no = 1
print("Dataset 3: 낙찰가 정보 수집 시작...")

while True:
    params = {
        "serviceKey": SERVICE_KEY, "pageNo": page_no, "numOfRows": numOfRows,
        "inqryDiv": 1, "inqryBgnDt": inqryBgnDt, "inqryEndDt": inqryEndDt, "type": "json"
    }
    try:
        res = session.get(url_scsbid, params=params).json()
        body = res.get("response", {}).get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict): items = items.get("item", [])
        if not isinstance(items, list): items = [items] if items else []
        if not items: break

        for item in items:
            scsbid_rows.append({
                "공고번호": item.get("bidNtceNo"),
                "차수": item.get("bidNtceOrd"),
                "낙찰가": item.get("sucsfbidAmt")
            })

        if len(scsbid_rows) >= body.get("totalCount", 0): break
        page_no += 1
    except:
        break

df_scsbid = pd.DataFrame(scsbid_rows)
df = df_all.merge(df_scsbid, on=["공고번호", "차수"], how="left")

# 저장
df.to_csv(filename, index=False, encoding="utf-8-sig")
print(f"🎉 모든 작업 완료! 파일 저장됨: {filename}")