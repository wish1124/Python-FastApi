import pandas as pd
import glob
import os
import numpy as np

def merge_csv_files(input_folder='./data', output_file='merged_output.csv'):
    """
    1단계: 지정된 폴더 내의 모든 CSV 파일을 병합합니다.
    """
    print(f"\n[1단계] CSV 파일 병합 시작 (폴더: {input_folder})")
    
    file_pattern = os.path.join(input_folder, '*.csv')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"❌ '{input_folder}' 경로에서 CSV 파일을 찾을 수 없습니다.")
        return False

    dfs = []
    for file in file_list:
        try:
            # low_memory=False: 대용량 파일 경고 방지
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
            # print(f" - 로드: {os.path.basename(file)} ({len(df)}행)")
        except Exception as e:
            print(f"❌ 에러 발생 ({file}): {e}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 병합 완료: {output_file} (총 {len(merged_df)}행)")
        return True
    else:
        print("❌ 병합할 데이터가 없습니다.")
        return False

def remove_duplicates(input_file='merged_output.csv', output_file='final_unique_data.csv'):
    """
    2단계: 병합된 파일에서 중복 행을 제거합니다.
    """
    print(f"\n[2단계] 중복 데이터 제거 시작")
    
    if not os.path.exists(input_file):
        print(f"❌ 파일이 없습니다: {input_file}")
        return False

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='cp949', low_memory=False)
        
    original_count = len(df)
    
    # 중복 제거
    df_unique = df.drop_duplicates()
    new_count = len(df_unique)
    
    df_unique.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f" - 제거된 중복 행: {original_count - new_count}개")
    print(f"✅ 중복 제거 완료: {output_file} (최종 {new_count}행)")
    return True

def preprocess_data(input_file='final_unique_data.csv', output_file='preprocessed_dataset.csv'):
    """
    3단계: 결측치 처리 및 데이터 정제
    - 낙찰가 NaN 제거
    - 낙찰하한율 NaN -> 최빈값(mode)으로 채우기
    - 추정가격 NaN 제거
    """
    print(f"\n[3단계] 데이터 전처리 시작")
    
    if not os.path.exists(input_file):
        print(f"❌ 파일이 없습니다: {input_file}")
        return

    df = pd.read_csv(input_file, low_memory=False)
    initial_count = len(df)

    # 1. 낙찰가가 없는 행 제거
    df = df.dropna(subset=['낙찰가'])
    print(f" - '낙찰가' NaN 제거 후: {len(df)}행")

    # 2. 낙찰하한율 결측치 채우기 (최빈값 사용)
    # 주의: mode()는 Series를 반환하므로 [0]으로 스칼라 값을 추출해야 함
    if df['낙찰하한율'].isnull().sum() > 0:
        mode_value = df['낙찰하한율'].mode()[0]
        df['낙찰하한율'] = df['낙찰하한율'].fillna(mode_value)
        print(f" - '낙찰하한율' 결측치 채움 (최빈값: {mode_value})")

    # 3. 추정가격 없는 행 제거
    df = df.dropna(subset=['추정가격'])
    print(f" - '추정가격' NaN 제거 후: {len(df)}행")

    # 최종 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"✅ 모든 전처리 완료!")
    print(f"📂 최종 저장 파일: {output_file}")
    print(f"📊 최종 데이터 크기: {len(df)}행 (삭제된 행: {initial_count - len(df)}개)")

if __name__ == "__main__":
    # 데이터 폴더 경로 설정 (필요시 수정)
    DATA_DIR = "./data" 
    
    # 파이프라인 실행
    if merge_csv_files(input_folder=DATA_DIR):
        if remove_duplicates():
            preprocess_data()
