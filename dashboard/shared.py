import pandas as pd
from pathlib import Path

class RealTimeStreamer:
    def __init__(self, data: pd.DataFrame):
        self.full_data = data.reset_index(drop=True).copy()
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
