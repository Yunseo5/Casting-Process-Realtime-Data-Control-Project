import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import time

class RealTimeStreamer:
    def __init__(self):
        # ✅ 상대 경로 사용
        BASE_DIR = Path(__file__).resolve().parents[1]
        csv_path = BASE_DIR / "data" / "raw" / "test.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            raise ValueError("CSV 파일에 'time' 컬럼이 없습니다.")

        start_time = "2019-03-18"
        t = pd.to_datetime(df["time"], errors="coerce")
        df["__date__"] = t.dt.date
        start_date = datetime.strptime(start_time, "%Y-%m-%d").date()
        mask = df["__date__"] >= start_date
        self.full_data = df.loc[mask].drop(columns="__date__", errors="ignore").reset_index(drop=True)

        self.current_index = 0
        self.interval_sec = 1.0
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_update_timestamp = 0

    def _background_update_loop(self):
        """정확한 1초 간격으로 데이터 진행"""
        next_update_time = time.time()
        
        while not self._stop_event.is_set():
            now = time.time()
            
            # 정확히 1초 경과했는지 확인
            if now >= next_update_time:
                with self._lock:
                    if self.is_running and self.current_index < len(self.full_data):
                        self.current_index += 1
                        self._last_update_timestamp = now
                
                next_update_time = now + self.interval_sec
            
            time.sleep(0.01)  # CPU 부하 줄임

    def get_current_data(self):
        with self._lock:
            if self.current_index == 0:
                return pd.DataFrame()
            return self.full_data.iloc[: self.current_index].copy()

    def get_last_update_time(self):
        with self._lock:
            return self.current_index

    def get_last_update_timestamp(self):
        """마지막 업데이트 시간 반환"""
        with self._lock:
            return self._last_update_timestamp

    def start_stream(self):
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._background_update_loop, daemon=True)
            self._thread.start()

    def stop_stream(self):
        self.is_running = False

    def reset_stream(self):
        with self._lock:
            self.current_index = 0
            self._last_update_timestamp = 0

    def cleanup(self):
        self.stop_stream()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def progress(self) -> float:
        with self._lock:
            if len(self.full_data) == 0:
                return 0.0
            return (self.current_index / len(self.full_data)) * 100