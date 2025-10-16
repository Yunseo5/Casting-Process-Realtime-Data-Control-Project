import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v1_time.csv"
OUTPUT_FILE_TEST = BASE_DIR / "data" / "processed" / "test_v1_time.csv"


# 데이터 로드
df = pd.read_csv(DATA_FILE)

# 데이터 정보
df.info()
df.columns
df.isna().sum()

# 데이터 
# 데이터가 이미 시간 순으로 정렬되어 있다고 가정
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ==================================================================================================
# date, time 컬럼명 swap 및 타입 변환
# ==================================================================================================
train_df = train_df.rename(columns={'date': '__tmp_swap__'})
train_df = train_df.rename(columns={'time': 'date', '__tmp_swap__': 'time'})

train_df["date"] = pd.to_datetime(train_df["date"], format="%Y-%m-%d")
train_df["time"] = pd.to_datetime(train_df["time"], format="%H:%M:%S")


# ==================================================================================================
# 대부분이 결측치인 행 확인 및 제거
# 해당 행이 유일한 emergency_stop 결측행이여서 이 행이 긴급중단을 나타내는 행이라고 판단
# 모델 예측 끝난 후에 ‘emergency_stop’이 결측인 경우 무조건 불량이라고 판정 내도록 만들기
# ==================================================================================================
train_df.iloc[19327, :]
mold_code_19327 = train_df.loc[19327, "mold_code"]
time_19327 = train_df.loc[19327, "time"]
train_df.loc[(train_df["mold_code"] == mold_code_19327) & (train_df["time"] == time_19327) & (train_df["id"] > 19273), :]
train_df.drop(19327, inplace=True)

# ==================================================================================================
# 단일값 컬럼 및 불필요한 컬럼 제거
# ==================================================================================================
# ID 컬럼 제거
train_df.drop(columns=["id"], inplace=True)
# 단일값 컬럼 제거
train_df["line"].unique()
train_df["name"].unique()
train_df["mold_name"].unique()
train_df.drop(columns=["line", "name", "mold_name"])
# nan값이 한개의 행인 emergency_stop 컬럼 제거 
train_df.drop(columns=["emergency_stop"], inplace=True)
# 중복 컬럼 제거
train_df.drop(columns=["registration_time"], inplace=True)

# ==================================================================================================
# 데이터가 겹치는 행 제거
# mold_code가 같으면서 count가 연속적으로 같은 행 제거
# ==================================================================================================
# mold_code별로 데이터 프레임 나누기
mold_codes = train_df["mold_code"].unique()
df_8722 = train_df[train_df["mold_code"] == 8722].copy()
df_8412 = train_df[train_df["mold_code"] == 8412].copy()
df_8573 = train_df[train_df["mold_code"] == 8573].copy()
df_8917 = train_df[train_df["mold_code"] == 8917].copy()
df_8600 = train_df[train_df["mold_code"] == 8600].copy()

# 연속된 count 행 제거 함수
def remove_consecutive_counts(df):
    prev_count = 0
    index_list = []

    for idx, row in df.iterrows():
        if row["count"] == prev_count:
            index_list.append(idx)
        prev_count = row["count"]

    df.drop(index=index_list, inplace=True)
    return df

df_8722 = remove_consecutive_counts(df_8722)
df_8412 = remove_consecutive_counts(df_8412)
df_8573 = remove_consecutive_counts(df_8573)
df_8917 = remove_consecutive_counts(df_8917)
df_8600 = remove_consecutive_counts(df_8600)

# 나눈 데이터 프레임 병합
train_df = pd.concat([df_8722, df_8412, df_8573, df_8917, df_8600])
# 인덱스 정렬
train_df = train_df.sort_index()


# ==================================================================================================
# 결측치 처리 (molten_temp)
# 처리 방법 : 동일코드 앞 생산 온도, 동일 코드 뒤 생산 온도 평균
# ==================================================================================================
# 원본 molten_temp를 새로운 열로 복사
train_df['molten_temp_filled'] = train_df['molten_temp']

# 코드별 시간 순 정렬 후 선형 보간
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)

# 채워진 컬럼으로 교체
train_df.drop(columns=["molten_temp"], inplace=True)
train_df = train_df.rename(columns={'molten_temp_filled': 'molten_temp'})

# ==================================================================================================
# 컬럼 제거 (upper_mold_temp3)
# 결측치 총 312개, 이상치(1449.0) 64356개로 정보값 매우 왜곡
# upper_mold_temp3가 결측일 때 mold_code_8412, lower_mold_temp3, molten_volume도 결측
# 이상치가 1449.0으로 고정이라서 센서가 고장났을 경우 1449라는 코드를 내보내는 것으로 가정하고 upper_mold_temp3 열을 제거하기로 함
# ==================================================================================================
train_df.drop(columns=["upper_mold_temp3"], inplace=True)

# ==================================================================================================
# 컬럼 제거 (lower_mold_temp3)
# 이상치(1449.0) 71651개, 결측치 312개로 upper_mold_temp3와 마찬가지로 제거하기로 함
# ==================================================================================================
train_df.drop(columns=["lower_mold_temp3"], inplace=True)

# ==================================================================================================
# 컬럼 제거 (heating_furnace)
# 결측치 총 40880개 (mold_code 8600은 전부 다 결측치(2960개), 8722도 전부 다 결측치(19664개))
# 일단은 제외 (3개 이상의 종류이지만 구분이 어려움, 결과에 큰 영향을 미치지 않을 것이라 판단)
# ==================================================================================================
train_df.drop(columns=["heating_furnace"], inplace=True)

# ==================================================================================================
# 컬럼 제거 (molten_volume)
# ==================================================================================================
train_df.drop(columns=["molten_volume"], inplace=True)

# ==================================================================================================
# 이상치 제거 (upper_mold_temp2)
# ==================================================================================================
train_df['upper_mold_temp2'].hist()
train_df['upper_mold_temp2'].describe()
train_df[train_df['upper_mold_temp2']==4232]
train_df.drop(42632,inplace=True)

# ==================================================================================================
# 행 & 열 제거 (tryshot_signal)
# 시범 운행이기 때문에 불량율 100퍼센트 -> 학습에 필요없는 데이터로 판단
# 모델 예측 끝난 후에 ‘tryshot_signal’이 D인 경우 무조건 불량이라고 판정 내도록 만들기
# ==================================================================================================
train_df = train_df[~(train_df["tryshot_signal"] == 'D')]
train_df.drop(columns=["tryshot_signal"], inplace=True)

train_df.to_csv(OUTPUT_FILE_TRAIN)
test_df.to_csv(OUTPUT_FILE_TEST)