# dashboard/modules/tab_target_ai_engineer.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from shiny import ui, render
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------
# 파일 경로 및 상수
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
LIGHTGBM_MODEL = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"
TEST_FEATURE_FILE = BASE_DIR / "data" / "raw" / "test.csv"
TEST_LABEL_FILE = BASE_DIR / "data" / "raw" / "test_label.csv"
TARGET_COLUMN = "passorfail"
DROP_COLUMNS = ["date", "time", "Unnamed: 0"]
CLASS_LABELS = ["정상", "불량"]
DEFAULT_FONT = "Malgun Gothic"


def _configure_fonts() -> None:
    # ensure Korean characters render; fallback silently if font missing
    try:
        font_manager.findfont(DEFAULT_FONT, fallback_to_default=False)
        plt.rcParams["font.family"] = DEFAULT_FONT
    except (ValueError, RuntimeError):
        pass
    plt.rcParams["axes.unicode_minus"] = False


_configure_fonts()


@lru_cache(maxsize=1)
def _load_artifact() -> dict:
    if not LIGHTGBM_MODEL.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {LIGHTGBM_MODEL}")
    return joblib.load(LIGHTGBM_MODEL)


@lru_cache(maxsize=1)
def _model_components():
    artifact = _load_artifact()
    model = artifact["model"]
    scaler = artifact.get("scaler")
    ordinal_encoder = artifact.get("ordinal_encoder")
    onehot_encoder = artifact.get("onehot_encoder")
    threshold = float(artifact.get("operating_threshold", 0.5))
    return model, scaler, ordinal_encoder, onehot_encoder, threshold


@lru_cache(maxsize=1)
def _numeric_columns() -> Tuple[str, ...]:
    _, scaler, _, _, _ = _model_components()
    if scaler is None or not hasattr(scaler, "feature_names_in_"):
        return tuple()
    return tuple(scaler.feature_names_in_)


@lru_cache(maxsize=1)
def _categorical_columns() -> Tuple[str, ...]:
    _, _, ordinal_encoder, _, _ = _model_components()
    if ordinal_encoder is None or not hasattr(ordinal_encoder, "feature_names_in_"):
        return tuple()
    return tuple(ordinal_encoder.feature_names_in_)


@lru_cache(maxsize=1)
def _load_test_dataframe() -> pd.DataFrame:
    if not TEST_FEATURE_FILE.exists():
        raise FileNotFoundError(f"테스트 데이터 파일이 없습니다: {TEST_FEATURE_FILE}")
    if not TEST_LABEL_FILE.exists():
        raise FileNotFoundError(f"테스트 라벨 파일이 없습니다: {TEST_LABEL_FILE}")

    features = pd.read_csv(TEST_FEATURE_FILE)
    labels = pd.read_csv(TEST_LABEL_FILE)

    if "id" not in features.columns or "id" not in labels.columns:
        raise KeyError("테스트 데이터와 라벨에 공통 'id' 컬럼이 필요합니다.")

    df = features.merge(labels, on="id", how="left", validate="one_to_one")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"라벨 파일에 '{TARGET_COLUMN}' 컬럼이 존재하지 않습니다.")

    df = df.copy()
    return df


@lru_cache(maxsize=1)
def _test_datetime_series() -> pd.Series:
    df = _load_test_dataframe()
    if "date" in df.columns and "time" in df.columns:
        timestamp = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )
    elif "registration_time" in df.columns:
        timestamp = pd.to_datetime(df["registration_time"], errors="coerce")
    else:
        raise KeyError("기간 설정을 위한 'date'/'time' 또는 'registration_time' 컬럼이 필요합니다.")
    timestamp.name = "timestamp"
    return timestamp


@lru_cache(maxsize=1)
def _load_test_frame() -> Tuple[pd.DataFrame, pd.Series]:
    df = _load_test_dataframe()
    df_model = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = df_model[TARGET_COLUMN].astype("Int64").fillna(0).astype(int)
    X = df_model.drop(columns=[TARGET_COLUMN])
    return X, y


def _booster_iteration(model) -> int | None:
    iteration = getattr(model, "best_iteration", None)
    if not iteration:
        current_iteration = getattr(model, "current_iteration", None)
        if callable(current_iteration):
            iteration = current_iteration()
    return int(iteration) if iteration else None


def _transform_features(X: pd.DataFrame) -> np.ndarray:
    _, scaler, ordinal_encoder, onehot_encoder, _ = _model_components()
    num_cols = list(_numeric_columns())
    cat_cols = list(_categorical_columns())
    row_count = len(X)

    if num_cols:
        missing_num = [col for col in num_cols if col not in X.columns]
        if missing_num:
            raise KeyError(f"입력 데이터에 수치형 컬럼 {missing_num} 이(가) 없습니다.")
        X_num = scaler.transform(X.loc[:, num_cols]) if scaler is not None else np.empty((row_count, 0))
    else:
        X_num = np.empty((row_count, 0))

    if cat_cols:
        if ordinal_encoder is None or onehot_encoder is None:
            raise RuntimeError("범주형 인코더가 누락되어 PDP를 계산할 수 없습니다.")
        missing_cat = [col for col in cat_cols if col not in X.columns]
        if missing_cat:
            raise KeyError(f"입력 데이터에 범주형 컬럼 {missing_cat} 이(가) 없습니다.")
        X_cat_ord = ordinal_encoder.transform(X.loc[:, cat_cols]).astype(int)
        X_cat_ohe = onehot_encoder.transform(X_cat_ord)
    else:
        X_cat_ohe = np.empty((row_count, 0))

    if X_num.size and X_cat_ohe.size:
        return np.hstack([X_num, X_cat_ohe])
    if X_num.size:
        return X_num
    if X_cat_ohe.size:
        return X_cat_ohe
    return np.zeros((row_count, 0))


def _predict_proba(X: pd.DataFrame) -> np.ndarray:
    model, _, _, _, _ = _model_components()
    matrix = _transform_features(X)
    iteration = _booster_iteration(model)
    if iteration:
        return model.predict(matrix, num_iteration=iteration)
    return model.predict(matrix)


@lru_cache(maxsize=1)
def _model_predictions() -> Tuple[np.ndarray, np.ndarray]:
    _, _, _, _, threshold = _model_components()
    X, y = _load_test_frame()

    probs = _predict_proba(X)
    preds = (probs >= threshold).astype(int)

    if "tryshot_signal" in X.columns:
        preds = preds.copy()
        preds[X["tryshot_signal"].to_numpy() == "D"] = 1

    return y.to_numpy(), preds


@lru_cache(maxsize=1)
def _model_evaluation() -> Tuple[np.ndarray, pd.DataFrame]:
    y_true, y_pred = _model_predictions()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = pd.DataFrame(
        [
            {"지표": "정확도", "값": accuracy_score(y_true, y_pred)},
            {"지표": "정밀도", "값": precision_score(y_true, y_pred, zero_division=0)},
            {"지표": "재현율", "값": recall_score(y_true, y_pred, zero_division=0)},
            {"지표": "F1-score", "값": f1_score(y_true, y_pred, zero_division=0)},
        ]
    )
    metrics["값"] = metrics["값"].map(lambda x: f"{x:.3f}")
    return cm, metrics


# PDP에서는 수치형 컬럼만 사용
PDP_FEATURES = _numeric_columns()


def _date_range_defaults() -> Tuple[str | None, str | None]:
    try:
        ts = _test_datetime_series().dropna()
        if ts.empty:
            return None, None
        start = ts.min().date().isoformat()
        end = ts.max().date().isoformat()
        return start, end
    except (FileNotFoundError, KeyError):
        return None, None


DATE_RANGE_START, DATE_RANGE_END = _date_range_defaults()


def _compute_pdp(feature: str, grid_size: int = 20) -> Tuple[pd.DataFrame, bool]:
    X, _ = _load_test_frame()
    if feature not in X.columns:
        raise KeyError(f"PDP를 계산할 수 없습니다. '{feature}' 컬럼이 존재하지 않습니다.")

    series = X[feature]
    if series.dropna().empty:
        raise ValueError(f"'{feature}' 컬럼에 유효한 값이 없어 PDP를 계산할 수 없습니다.")

    working = X.copy()
    if pd.api.types.is_numeric_dtype(series):
        lower, upper = np.nanpercentile(series, [1, 99])
        if np.isclose(lower, upper):
            values = np.array([series.dropna().median()])
        else:
            values = np.linspace(lower, upper, grid_size)
        probabilities = []
        for val in values:
            working[feature] = val
            probabilities.append(float(np.mean(_predict_proba(working))))
        pdp_df = pd.DataFrame({"feature_value": values, "probability": probabilities})
        return pdp_df, True

    categories = sorted(series.dropna().unique())
    probabilities = []
    for cat in categories:
        working[feature] = cat
        probabilities.append(float(np.mean(_predict_proba(working))))
    pdp_df = pd.DataFrame({"feature_value": categories, "probability": probabilities})
    return pdp_df, False


@lru_cache(maxsize=1)
def _prediction_summary() -> pd.DataFrame:
    X, y = _load_test_frame()
    df_full = _load_test_dataframe()
    timestamps = _test_datetime_series()

    model, _, ordinal_encoder, onehot_encoder, threshold = _model_components()
    matrix = _transform_features(X)
    iteration = _booster_iteration(model)

    probs = model.predict(matrix, num_iteration=iteration)
    preds = (probs >= threshold).astype(int)

    if "tryshot_signal" in X.columns:
        tryshot = X["tryshot_signal"].to_numpy()
        mask = pd.Series(tryshot == "D")
        if mask.any():
            preds = preds.copy()
            preds[mask.to_numpy()] = 1

    contrib = model.predict(matrix, num_iteration=iteration, pred_contrib=True)
    feature_contrib = contrib[:, :-1]  # drop bias term

    per_feature: dict[str, np.ndarray] = {}
    pointer = 0
    numeric_cols = list(_numeric_columns())
    categorical_cols = list(_categorical_columns())

    for col in numeric_cols:
        per_feature[col] = feature_contrib[:, pointer]
        pointer += 1

    if categorical_cols:
        if onehot_encoder is None:
            raise RuntimeError("원-핫 인코더가 없어 범주형 기여도를 계산할 수 없습니다.")
        for col, categories in zip(categorical_cols, onehot_encoder.categories_):
            count = len(categories)
            if count == 0:
                per_feature[col] = np.zeros(len(X))
            else:
                per_feature[col] = feature_contrib[:, pointer : pointer + count].sum(axis=1)
            pointer += count

    remaining = feature_contrib.shape[1] - pointer
    if remaining > 0:
        per_feature["기타"] = feature_contrib[:, pointer:].sum(axis=1)

    if not per_feature:
        per_feature_df = pd.DataFrame(index=X.index)
    else:
        per_feature_df = pd.DataFrame(per_feature, index=X.index)

    if per_feature_df.empty:
        top_feature = pd.Series(data=pd.NA, index=X.index, name="top_feature")
        top_value = pd.Series(data=np.nan, index=X.index, name="top_value")
    else:
        top_feature = per_feature_df.idxmax(axis=1)
        top_value = per_feature_df.max(axis=1)

    ids = df_full["id"].to_numpy() if "id" in df_full.columns else np.arange(len(X))
    summary = pd.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps.to_numpy(),
            "probability": probs,
            "prediction": preds,
            "actual": y.to_numpy() if isinstance(y, pd.Series) else y,
            "top_feature": top_feature,
            "top_value": top_value,
        }
    )
    return summary


def _top_feature_counts(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    summary = _prediction_summary()
    if "timestamp" not in summary.columns:
        raise KeyError("타임스탬프 정보가 없어 기간 필터링을 할 수 없습니다.")

    df = summary.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = ts.notna()

    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask &= ts >= start_dt
    if end_date:
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        mask &= ts <= end_dt

    df = df.loc[mask & (df["prediction"] == 1)]
    df = df[df["top_feature"].notna()]

    if df.empty:
        return pd.DataFrame(
            [
                {"변수": "데이터 없음", "건수": 0, "비율": "0.0%"}
            ]
        )

    counts = df["top_feature"].value_counts().rename_axis("변수").reset_index(name="건수")
    total = counts["건수"].sum()
    counts["비율"] = (counts["건수"] / total * 100).map(lambda x: f"{x:.1f}%")
    return counts


# 탭별 UI ----------------------------------------------------------------------
tab_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel(
            "MODEL",
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("혼동행렬"),
                    ui.output_plot("plot_model_confusion", height="360px"),
                ),
                ui.card(
                    ui.card_header("성능지표"),
                    ui.output_table("table_model_metrics"),
                ),
                width=1 / 2,
                gap="1rem",
            ),
        ),
        ui.nav_panel(
            "Interpretation",
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Partial Dependence (PDP)"),
                    ui.input_select(
                        "pdp_feature",
                        "변수 선택",
                        choices=list(PDP_FEATURES),
                        selected=PDP_FEATURES[0] if PDP_FEATURES else None,
                    ),
                    ui.output_plot("plot_pdp_feature", height="360px"),
                ),
                ui.card(
                    ui.card_header("불량 영향 변수 비율"),
                    ui.input_date_range(
                        "failure_period",
                        "기간 설정",
                        start=DATE_RANGE_START,
                        end=DATE_RANGE_END,
                        min=DATE_RANGE_START,
                        max=DATE_RANGE_END,
                    ),
                    ui.output_plot("plot_failure_feature_share", height="360px"),
                ),
                width=1 / 2,
                fill=True,
                gap="1rem",
            ),
        ),
    ),
)


# 탭별 서버 --------------------------------------------------------------------
def tab_server(input, output, session):
    @render.plot
    def plot_model_confusion():
        fig, ax = plt.subplots(figsize=(4, 3.8))
        try:
            cm, _ = _model_evaluation()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], CLASS_LABELS)
            ax.set_yticks([0, 1], CLASS_LABELS)
            ax.set_xlabel("예측")
            ax.set_ylabel("실제")
            for (i, j), value in np.ndenumerate(cm):
                ax.text(j, i, int(value), ha="center", va="center", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
        except (FileNotFoundError, KeyError) as err:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                str(err),
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
        return fig

    @render.table
    def table_model_metrics():
        try:
            _, metrics = _model_evaluation()
            return metrics
        except (FileNotFoundError, KeyError) as err:
            return pd.DataFrame([{"지표": "오류", "값": str(err)}])

    @render.plot
    def plot_pdp_feature():
        feature = input.pdp_feature()
        fig, ax = plt.subplots(figsize=(6.5, 3.6))

        if not feature:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "변수를 선택하면 PDP가 표시됩니다.",
                ha="center",
                va="center",
                fontsize=11,
            )
            return fig

        try:
            pdp_df, is_numeric = _compute_pdp(feature)
            if is_numeric:
                ax.plot(
                    pdp_df["feature_value"],
                    pdp_df["probability"],
                    color="#2A9D8F",
                    marker="o",
                )
                ax.fill_between(
                    pdp_df["feature_value"],
                    pdp_df["probability"],
                    color="#2A9D8F",
                    alpha=0.15,
                )
            else:
                categories = pdp_df["feature_value"].astype(str)
                ax.bar(categories, pdp_df["probability"], color="#E76F51", alpha=0.85)
                plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
                for idx, prob in enumerate(pdp_df["probability"]):
                    ax.text(
                        idx,
                        prob,
                        f"{prob:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            ax.set_ylim(0, 1)
            ax.set_ylabel("불량 확률")
            ax.set_xlabel(feature)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as err:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                str(err),
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
        return fig

    @render.plot
    def plot_failure_feature_share():
        start, end = input.failure_period()
        fig, ax = plt.subplots(figsize=(4, 3.6))
        try:
            counts_df = _top_feature_counts(start, end)
            if counts_df.empty or counts_df["건수"].sum() == 0 or (
                len(counts_df) == 1 and counts_df.iloc[0]["변수"] == "데이터 없음"
            ):
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "선택한 기간에 불량 예측이 없습니다.",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                return fig

            y_labels = counts_df["변수"]
            x_values = counts_df["건수"]
            ax.barh(y_labels, x_values, color="#457b9d")
            for y, x, ratio in zip(y_labels, x_values, counts_df["비율"]):
                ax.text(
                    x + max(x_values) * 0.02,
                    y,
                    f"{x} ({ratio})",
                    va="center",
                    fontsize=10,
                )
            ax.set_xlabel("건수")
            ax.set_ylabel("변수")
            ax.grid(alpha=0.3, axis="x")
            ax.set_xlim(0, max(x_values) * 1.08)
            fig.subplots_adjust(left=0.42, right=0.98, top=0.95, bottom=0.15)
        except (FileNotFoundError, KeyError, RuntimeError) as err:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                str(err),
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
        return fig
