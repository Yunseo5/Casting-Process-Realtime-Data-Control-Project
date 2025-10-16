import pandas as pd
from pathlib import Path
from datetime import date

class RealTimeStreamer:
    def __init__(
        self,
        csv_path: str | Path = r"C:/Users/USER/Desktop/Casting-Process-Realtime-Data-Control-Project/data/test.csv",
        start_time: str = "2019-03-18",
        include_after: bool = True,  # True면 해당 날짜 '이후' 포함, False면 '해당 날짜만'
    ):
        # 1) CSV 로드
        self.full_data = pd.read_csv(csv_path)

        if "time" not in self.full_data.columns:
            raise ValueError("CSV 파일에 'time' 컬럼이 없습니다.")

        # 2) datetime으로 안전하게 파싱 (공백/형식/시간 포함 대응)
        t = pd.to_datetime(self.full_data["time"], errors="coerce", utc=False)
        if t.isna().all():
            raise ValueError("time 컬럼을 datetime으로 파싱하지 못했습니다. 원본 형식을 확인하세요.")
        self.full_data["__date__"] = t.dt.date  # 날짜만 보관

        # 3) 시작 날짜 설정
        start_date: date = pd.to_datetime(start_time).date()

        # 4) 필터링 (해당 날짜 '이후' 또는 '그 날짜만')
        if include_after:
            mask = self.full_data["__date__"] >= start_date
        else:
            mask = self.full_data["__date__"] == start_date

        if not mask.any():
            # 디버깅 도움: CSV에 들어있는 날짜 범위를 보여줌
            dmin, dmax = self.full_data["__date__"].min(), self.full_data["__date__"].max()
            raise ValueError(
                f"시작 날짜 {start_date}에 해당하는 행이 없습니다. "
                f"CSV 날짜 범위: {dmin} ~ {dmax}"
            )

        # 5) 시작 구간으로 자르고 인덱스 초기화
        self.full_data = self.full_data.loc[mask].reset_index(drop=True).drop(columns="__date__", errors="ignore")
        self.current_index = 0

    def get_next_batch(self, batch_size: int = 1):
        if self.current_index >= len(self.full_data):
            return None
        end_index = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[: self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0

    def progress(self) -> float:
        if len(self.full_data) == 0:
            return 0.0
        return (self.current_index / len(self.full_data)) * 100
