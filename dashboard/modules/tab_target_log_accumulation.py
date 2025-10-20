from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple

# LightGBM 모델 로드
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"

# ---------------------------
# 모델 및 전처리기 로드
# ---------------------------
try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    ordinal_encoder = artifact.get("ordinal_encoder")
    onehot_encoder = artifact.get("onehot_encoder")
    threshold = float(artifact.get("operating_threshold", 0.5))

    # 학습 시점 피처 스키마(수치/범주) 고정
    numeric_cols: Tuple[str, ...] = tuple(getattr(scaler, "feature_names_in_", ())) if scaler is not None else tuple()
    categorical_cols: Tuple[str, ...] = tuple(getattr(ordinal_encoder, "feature_names_in_", ())) if ordinal_encoder is not None else tuple()

    MODEL_LOADED = True
except Exception as e:
    print(f"모델 로드 실패: {e}")
    MODEL_LOADED = False
    model = None
    scaler = None
    ordinal_encoder = None
    onehot_encoder = None
    threshold = 0.5
    numeric_cols = tuple()
    categorical_cols = tuple()

# ---------------------------
# 드롭할 컬럼(원시 로그 컬럼들)
# ---------------------------
DROP_COLUMNS = [
    "line", "name", "mold_name", "date", "time", "Unnamed: 0", "id",
]

# ---------------------------
# LightGBM 예측 유틸
# ---------------------------

def _booster_iteration(model) -> int | None:
    """LightGBM best_iteration 안전 추출."""
    iteration = getattr(model, "best_iteration", None)
    if not iteration:
        # lightgbm.Booster의 current_iteration() 지원
        current_iteration = getattr(model, "current_iteration", None)
        if callable(current_iteration):
            iteration = current_iteration()
    return int(iteration) if iteration else None


def _transform_features(X: pd.DataFrame) -> np.ndarray:
    """학습 시점 스키마에 맞춰 수치/범주 인코딩 조합.

    - 수치형: 학습에 사용된 numeric_cols 순서대로 scaler.transform
    - 범주형: 학습에 사용된 categorical_cols -> ordinal -> onehot
    - 학습 스키마에 있는 컬럼이 누락되면 오류(침묵 드랍 방지)
    """
    row_count = len(X)

    # 수치형
    if numeric_cols:
        missing_num = [c for c in numeric_cols if c not in X.columns]
        if missing_num:
            raise KeyError(f"입력 데이터에 수치형 컬럼 {missing_num} 이(가) 없습니다.")
        if scaler is None:
            X_num = np.empty((row_count, 0))
        else:
            X_num = scaler.transform(X.loc[:, list(numeric_cols)])
    else:
        X_num = np.empty((row_count, 0))

    # 범주형
    if categorical_cols:
        if ordinal_encoder is None or onehot_encoder is None:
            raise RuntimeError("범주형 인코더가 누락되어 예측을 수행할 수 없습니다.")
        missing_cat = [c for c in categorical_cols if c not in X.columns]
        if missing_cat:
            raise KeyError(f"입력 데이터에 범주형 컬럼 {missing_cat} 이(가) 없습니다.")
        X_cat_ord = ordinal_encoder.transform(X.loc[:, list(categorical_cols)]).astype(int)
        X_cat = onehot_encoder.transform(X_cat_ord)
    else:
        X_cat = np.empty((row_count, 0))

    # 결합
    if X_num.size and X_cat.size:
        return np.hstack([X_num, X_cat])
    if X_num.size:
        return X_num
    if X_cat.size:
        return X_cat
    # 모델이 입력을 요구하지만 스키마가 비어있는 경우
    return np.zeros((row_count, 0))


def _predict_proba(df: pd.DataFrame) -> np.ndarray:
    """원본 DF에서 불필요 컬럼 제거 → 스키마 기준 변환 → 확률 예측."""
    # 예측에 사용할 X 구성(원시 로그 컬럼 제거; 학습 스키마 컬럼들은 반드시 존재해야 함)
    X = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    features = _transform_features(X)
    iteration = _booster_iteration(model)
    if iteration:
        return model.predict(features, num_iteration=iteration)
    return model.predict(features)


def predict_passorfail(df: pd.DataFrame) -> np.ndarray:
    """데이터프레임에 대해 양품/불량 예측을 수행하여 0/1 배열 반환.

    규칙:
    - LightGBM 불량확률 >= threshold → 1(불량), 그 외 0(양품)
    - tryshot_signal == 'D' → 강제 1(불량)
    """
    if not MODEL_LOADED or df is None or df.empty:
        return np.zeros(0, dtype=int) if df is None else np.zeros(len(df), dtype=int)

    try:
        probs = _predict_proba(df)
        preds = (probs >= threshold).astype(int)

        # tryshot_signal 강제 불량 처리
        X_all = df.drop(columns=DROP_COLUMNS, errors="ignore")
        if "tryshot_signal" in X_all.columns:
            mask = X_all["tryshot_signal"].to_numpy() == "D"
            if np.any(mask):
                preds = preds.copy()
                preds[mask] = 1
        return preds

    except Exception as e:
        # 예측 실패 시 안전하게 0 반환(로그만 출력)
        print(f"예측 실패: {e}")
        return np.zeros(len(df), dtype=int)


# ---------------------------
# UI
# ---------------------------

tab_ui = ui.page_fluid(
    ui.div(
        # 통계 정보
        ui.div(
            ui.output_ui("tab_log_stats"),
            style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)",
        ),

        # DataGrid (전체 데이터)
        ui.div(
            ui.div(
                ui.HTML('<i class="fa-solid fa-table-list"></i>'),
                " 누적 데이터 (전체)",
                style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #e0e0e0",
            ),
            ui.output_ui("tab_log_table_wrapper"),
            style="background:#fff;border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)",
        ),

        style="max-width:1400px;margin:0 auto;padding:20px 0",
    ),
)


# ---------------------------
# Server
# ---------------------------

def tab_server(input, output, session, streamer, shared_df, streaming_active):

    @output
    @render.ui
    def tab_log_stats():
        df = shared_df.get()

        if df.empty:
            total_rows = 0
            memory_usage = "0 KB"
        else:
            total_rows = len(df)
            memory_usage = f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"

        return ui.div(
            ui.div(
                ui.HTML('<i class="fa-solid fa-list-ol"></i>'),
                f" 총 데이터 행: {total_rows:,}",
                style="font-weight:600;font-size:16px;color:#2c3e50",
            ),
            ui.div(
                ui.HTML('<i class="fa-solid fa-memory"></i>'),
                f" 메모리 사용량: {memory_usage}",
                style="font-weight:600;font-size:16px;color:#2c3e50;margin-top:10px",
            ),
        )

    @output
    @render.ui
    def tab_log_table_wrapper():
        df = shared_df.get()

        # 데이터가 없을 때 - 스크롤 없는 메시지
        if df.empty:
            return ui.div(
                ui.div(
                    "⏳ 데이터를 불러오는 중...",
                    style="text-align:center;padding:40px;color:#666;font-size:16px"
                ),
                style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9"
            )

        # 데이터가 있을 때 - DataGrid 표시
        return ui.div(
            ui.output_data_frame("tab_log_table_all"),
            style="width:100%;overflow-x:auto;overflow-y:auto;border:1px solid #e0e0e0;border-radius:8px;height:600px"
        )

    @output
    @render.data_frame
    def tab_log_table_all():
        df = shared_df.get()

        # 전체 데이터 복사
        result = df.copy()

        # passorfail 예측 추가(LightGBM 기준 + tryshot 강제 규칙)
        result["passorfail"] = predict_passorfail(result)

        # 불필요한 컬럼 제거(표시용)
        result = result.drop(columns=DROP_COLUMNS, errors='ignore')

        return render.DataGrid(
            result,
            height="600px",
            width="100%",
            filters=False,
            row_selection_mode="none",
        )
