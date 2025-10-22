from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = BASE_DIR / "data" / "intermin" / "test_predictions_v2.csv"
SEGMENT_COUNT = 15
X_AXIS_MARGIN = 1 / 3
THRESHOLD = 0.95
MOLD_CODES = ["8412", "8917", "8722", "8413", "8576"]
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_v1_time.csv"
MODEL_SCORE_FILE = BASE_DIR / "data" / "models" / "LightGBM_v2_scores.csv"
FEATURE_FILTERS: dict[str, dict[str, float]] = {
    "cast_pressure": {"min": 300.0},
    "low_section_speed": {"min": 75.0, "max": 150.0},
    "biscuit_thickness": {"max": 100.0},
    "high_section_speed": {"min": 75.0, "max": 175.0},
    "upper_mold_temp1": {"max": 400.0},
}

PI_TOP_FEATURES = pd.DataFrame({"feature": [
    "cast_pressure",
    "low_section_speed",
    "biscuit_thickness",
    "high_section_speed",
    "upper_mold_temp1",
]})




def _available_date_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if "timestamp" not in df.columns:
        return None, None
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None, None
    days = timestamps.dt.floor("D")
    return days.min(), days.max()


def _load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_FILE).reset_index(drop=True)
    if {"date", "time"}.issubset(df.columns):
        combined_dt = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        combined_td = df["time"].astype(str).str.strip() + " " + df["date"].astype(str).str.strip()
        ts = pd.to_datetime(combined_dt, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if ts.isna().all():
            ts = pd.to_datetime(combined_td, errors="coerce", format="%H:%M:%S %Y-%m-%d")
        fallback = pd.to_datetime(combined_dt, errors="coerce")
        ts = ts.where(~ts.isna(), fallback)
        ts = ts.where(~ts.isna(), pd.to_datetime(combined_td, errors="coerce"))
        df["timestamp"] = ts
    else:
        df["timestamp"] = pd.NaT
    return df


    df = df.copy()
    metric_series = df.get("metric")
    if metric_series is not None:
        df = df.loc[metric_series.astype(str).str.lower() == "f1"]

    if "importance_mean" in df.columns:
        df = df.sort_values("importance_mean", ascending=False)

    columns = [col for col in ["feature", "importance_mean"] if col in df.columns]
    if not columns:
        return pd.DataFrame(columns=["feature", "importance_mean"])

    df = df.loc[:, columns].head(max_items)
    if "importance_mean" not in df.columns:
        df["importance_mean"] = pd.NA
    return df.reset_index(drop=True)


def _load_train_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv(TRAIN_DATA_PATH)
    except Exception:
        return pd.DataFrame()


def _apply_feature_filters(values: pd.Series, feature: str) -> pd.Series:
    rule = FEATURE_FILTERS.get(feature)
    if not rule:
        return values
    filtered = values
    if "min" in rule:
        filtered = filtered[filtered >= rule["min"]]
    if "max" in rule:
        filtered = filtered[filtered <= rule["max"]]
    return filtered


def _segment_f1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "label": pd.Series(dtype="string"),
                "day": pd.Series(dtype="datetime64[ns]"),
                "f1": pd.Series(dtype=float),
            }
        )

    if "passorfail" not in df.columns or "prediction" not in df.columns:
        raise KeyError("예측 파일에 'passorfail' 또는 'prediction' 컬럼이 없습니다.")

    timestamps = pd.to_datetime(df.get("timestamp"), errors="coerce")
    valid_mask = timestamps.notna()
    valid_df = df.loc[valid_mask].copy()
    valid_times = timestamps.loc[valid_mask]

    if not valid_df.empty:
        valid_df["_day"] = valid_times.dt.floor("D")
        records: list[dict[str, object]] = []
        for day, subset in valid_df.groupby("_day", sort=True):
            if subset.empty:
                continue
            f1 = f1_score(subset["passorfail"].astype(int), subset["prediction"].astype(int), zero_division=0)
            records.append({"label": day.strftime("%Y-%m-%d"), "day": day, "f1": f1})
        if records:
            return pd.DataFrame(records)

    segments: list[dict[str, object]] = []
    indices = np.array_split(df.index.to_numpy(), SEGMENT_COUNT)
    for idx, idx_block in enumerate(indices, start=1):
        if len(idx_block) == 0:
            continue
        slice_df = df.loc[idx_block]
        f1 = f1_score(slice_df["passorfail"].astype(int), slice_df["prediction"].astype(int), zero_division=0)
        segments.append({"label": f"구간 {idx}", "day": pd.NaT, "f1": f1})

    return pd.DataFrame(segments)


PI_TOP_FEATURES = PI_TOP_FEATURES
TRAIN_DATASET = _load_train_dataset()


plot_card = ui.card(
    ui.card_header("금형 코드별 F1-Score 추이"),
    ui.output_plot("plot_segment_f1", height="420px"),
    ui.layout_columns(
        ui.output_plot("plot_confusion_matrix", height="260px"),
        ui.div(
            ui.output_data_frame("metrics_table"),
            class_="mt-2",
        ),
        col_widths=[6, 6],
    ),
)

buttons_card = ui.card(
    ui.card_header("금형 코드"),
    ui.output_ui("mold_buttons"),
    ui.div(
        ui.input_date_range("date_range", "기간 선택"),
        class_="mt-3",
    ),
    style="flex:2 1 0;width:100%;",
)

distribution_card = ui.card(
    ui.card_header("분포 분석"),
    ui.div(
        ui.input_action_button("btn_distribution_view", "분포확인", class_="btn btn-outline-secondary w-100"),
        class_="d-flex flex-column gap-2",
    ),
    style="flex:1 1 0;width:100%;",
)


plot_column = ui.div(plot_card, class_="h-100 d-flex flex-column w-100")

right_column = ui.div(
    buttons_card,
    distribution_card,
    class_="d-flex flex-column h-100 w-100",
)

tab_ui = ui.page_fluid(
    ui.layout_columns(
        plot_column,
        right_column,
        col_widths=[9, 3],
        class_="align-items-stretch",
    )
)


def tab_server(input, output, session):
    df_predictions = _load_predictions()
    if "mold_code" in df_predictions.columns:
        available_codes = set(df_predictions["mold_code"].dropna().astype(str))
    else:
        available_codes = set()
    mold_codes = [code for code in MOLD_CODES if code in available_codes]
    selected_mold = reactive.Value(mold_codes[0] if mold_codes else None)
    modal_selected_day = reactive.Value(None)

    @render.ui
    def mold_buttons():
        current = selected_mold.get()
        buttons = []
        codes_to_show = mold_codes if mold_codes else MOLD_CODES
        for code in codes_to_show:
            is_available = code in available_codes
            is_active = code == current
            classes = "btn w-100 mb-2 "
            styles = ""
            if is_active:
                classes += "text-white"
                styles = "background-color:#383636;border-color:#383636;"
            else:
                classes += "btn-outline-secondary"
            buttons.append(
                ui.input_action_button(
                    f"btn_mold_{code}",
                    code,
                    class_=classes.strip(),
                    disabled=not is_available,
                    style=styles or None,
                )
            )
        if not buttons:
            return ui.p("등록된 금형 코드가 없습니다.")
        return ui.div(*buttons, class_="d-flex flex-column")

    for code in MOLD_CODES:
        btn_id = f"btn_mold_{code}"

        @reactive.effect
        def _update_selected(code=code, btn_id=btn_id):
            btn = getattr(input, btn_id)
            if btn() and code in available_codes and selected_mold.get() != code:
                selected_mold.set(code)

    def _filtered_predictions_df() -> pd.DataFrame:
        filter_code = selected_mold.get()
        if filter_code is not None and filter_code in available_codes:
            df_filtered = df_predictions.loc[df_predictions["mold_code"].astype(str) == filter_code].copy()
        else:
            df_filtered = df_predictions.copy()

        date_window = _current_date_range()
        if date_window and "timestamp" in df_filtered.columns:
            start_day, end_day = date_window
            timestamps = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
            day_values = timestamps.dt.floor("D")
            mask = day_values.between(start_day, end_day)
            df_filtered = df_filtered.loc[mask]
        return df_filtered


    @reactive.effect
    def _sync_date_range_input():
        min_day, max_day = _available_date_bounds(df_predictions)
        if min_day is None or max_day is None:
            ui.update_date_range(
                "date_range",
                start=None,
                end=None,
                min=None,
                max=None,
            )
            return

        try:
            current_range = input.date_range()
        except Exception:
            current_range = None

        start_date = min_day.date()
        end_date = max_day.date()

        update_start = start_date
        update_end = end_date

        if current_range and len(current_range) == 2:
            cs, ce = current_range
            cs_dt = pd.to_datetime(cs, errors="coerce") if cs else pd.NaT
            ce_dt = pd.to_datetime(ce, errors="coerce") if ce else pd.NaT
            if pd.notna(cs_dt) and pd.notna(ce_dt):
                cs_day = cs_dt.date()
                ce_day = ce_dt.date()
                cs_day = min(max(cs_day, start_date), end_date)
                ce_day = min(max(ce_day, start_date), end_date)
                if ce_day < cs_day:
                    ce_day = cs_day
                update_start = cs_day
                update_end = ce_day

        ui.update_date_range(
            "date_range",
            start=update_start.isoformat(),
            end=update_end.isoformat(),
            min=start_date.isoformat(),
            max=end_date.isoformat(),
        )

    def _segment_with_flags() -> pd.DataFrame:
        global THRESHOLD
        segment_df = _segment_f1(_filtered_predictions_df())
        if segment_df.empty:
            return segment_df.assign(valid=False, below_threshold=False, threshold=THRESHOLD)

        f1_values = segment_df["f1"].to_numpy(dtype=float)
        valid_mask = (~np.isnan(f1_values)) & (f1_values > 0)
        model_threshold = _get_model_f1_score()
        daily_candidate: float | None = None
        if np.any(valid_mask):
            valid_values = f1_values[valid_mask]
            mean = float(np.nanmean(valid_values))
            std = float(np.nanstd(valid_values))
            if np.isfinite(mean) and np.isfinite(std):
                daily_candidate = float(np.clip(mean - std, 0.0, 1.0))

        candidates = [
            value
            for value in (daily_candidate, model_threshold)
            if value is not None and np.isfinite(value)
        ]
        dynamic_threshold = THRESHOLD
        if candidates:
            dynamic_threshold = float(np.clip(min(candidates), 0.0, 1.0))

        THRESHOLD = dynamic_threshold
        below_mask = valid_mask & (f1_values < dynamic_threshold)
        valid_series = pd.Series(valid_mask, index=segment_df.index)
        below_series = pd.Series(below_mask, index=segment_df.index)
        return segment_df.assign(
            valid=valid_series,
            below_threshold=below_series,
            threshold=dynamic_threshold,
        )

    def _highlighted_dates() -> pd.DataFrame:
        segment_df = _segment_with_flags()
        if segment_df.empty:
            return pd.DataFrame(columns=["날짜", "F1-score"])
        highlight_df = segment_df.loc[segment_df["below_threshold"], ["label", "f1"]].copy()
        highlight_df.columns = ["날짜", "F1-score"]
        highlight_df.sort_values("날짜", inplace=True)
        highlight_df["F1-score"] = highlight_df["F1-score"].astype(float).round(2)
        return highlight_df.reset_index(drop=True)


    def _current_date_range() -> tuple[pd.Timestamp, pd.Timestamp] | None:
        try:
            date_range = input.date_range()
        except Exception:
            return None
        if not date_range or len(date_range) != 2:
            return None
        start_value, end_value = date_range
        if start_value is None or end_value is None:
            return None
        start_ts = pd.to_datetime(start_value, errors="coerce")
        end_ts = pd.to_datetime(end_value, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            return None
        start_day = start_ts.floor("D")
        end_day = end_ts.floor("D")
        if end_day < start_day:
            start_day, end_day = end_day, start_day
        return start_day, end_day

    def _filtered_by_modal_day() -> tuple[pd.DataFrame, str | None]:
        df_filtered = _filtered_predictions_df()
        selected_value = modal_selected_day.get()
        if selected_value and "timestamp" in df_filtered.columns:
            timestamps = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
            df_filtered = df_filtered.loc[timestamps.dt.strftime("%Y-%m-%d") == selected_value]
        elif not selected_value:
            df_filtered = df_filtered.iloc[0:0]
        return df_filtered, selected_value

    def _highlighted_prediction_rows() -> pd.DataFrame:
        df_filtered = _filtered_predictions_df()
        if df_filtered.empty or "timestamp" not in df_filtered.columns:
            return df_filtered.iloc[0:0]

        highlight_df = _highlighted_dates()
        if highlight_df.empty:
            return df_filtered.iloc[0:0]

        highlight_days = highlight_df["날짜"].astype(str).tolist()
        timestamps = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        mask = timestamps.dt.strftime("%Y-%m-%d").isin(highlight_days)
        if not mask.any():
            return df_filtered.iloc[0:0]
        return df_filtered.loc[mask].copy()

    def _non_highlighted_prediction_rows() -> pd.DataFrame:
        df_filtered = _filtered_predictions_df()
        if df_filtered.empty:
            return df_filtered.iloc[0:0]
        highlight_df = _highlighted_prediction_rows()
        if highlight_df.empty:
            return df_filtered.copy()
        return df_filtered.drop(index=highlight_df.index, errors="ignore").copy()

    @render.plot
    def plot_segment_f1():
        segment_df = _segment_with_flags()
        fig, ax = plt.subplots(figsize=(7.2, 4.5))

        if segment_df.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha="center", va="center", fontsize=12)
            return fig

        segment_count = len(segment_df)
        x_positions = np.arange(1, segment_count + 1)
        f1_values = segment_df["f1"].to_numpy(dtype=float)
        valid_mask = segment_df["valid"].to_numpy(dtype=bool)
        below_mask = segment_df["below_threshold"].to_numpy(dtype=bool)

        line_color = "#383636"
        above_color = "#383636"
        below_color = "#FF0000"
        highlight_color = "#F9C5C5"

        for pos, is_low in zip(x_positions, below_mask):
            if is_low:
                left = pos - 0.5
                right = pos + 0.5
                ax.axvspan(left, right, color=highlight_color, alpha=0.3, zorder=0)

        above_mask = valid_mask & (~below_mask)
        if np.any(valid_mask):
            ax.plot(x_positions[valid_mask], f1_values[valid_mask], color=line_color, linewidth=2, zorder=2)
        if np.any(above_mask):
            ax.scatter(
                x_positions[above_mask],
                f1_values[above_mask],
                color=above_color,
                edgecolor="white",
                zorder=3,
            )
        if np.any(below_mask):
            ax.scatter(
                x_positions[below_mask],
                f1_values[below_mask],
                color=below_color,
                edgecolor="white",
                zorder=3,
            )

        valid_f1_mean = np.nan
        valid_f1_std = np.nan
        if np.any(valid_mask):
            valid_values = f1_values[valid_mask]
            valid_f1_mean = float(np.nanmean(valid_values))
            valid_f1_std = float(np.nanstd(valid_values))
        threshold_line = THRESHOLD
        if "threshold" in segment_df.columns:
            threshold_series = segment_df["threshold"].dropna()
            if not threshold_series.empty:
                threshold_line = float(threshold_series.iloc[0])
        ax.axhline(threshold_line, color="#C1121F", linestyle="--", linewidth=1.2)
        label_x = x_positions[0] - 0.45
        ax.text(
            label_x,
            threshold_line,
            f"{threshold_line:.2f}",
            fontsize=9,
            color="#C1121F",
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C1121F", alpha=0.75),
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("")
        ax.grid(alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xlim(0.5 - X_AXIS_MARGIN, segment_count + 0.5 + X_AXIS_MARGIN)

        labels = segment_df["label"].fillna("").astype(str).tolist()
        has_dates = segment_df.get("day") is not None and segment_df["day"].notna().any()
        if has_dates:
            highlight_labels = [label if is_low else "" for label, is_low in zip(labels, below_mask)]
            ax.set_xticklabels(highlight_labels, rotation=30, ha="right")
        else:
            ax.set_xticklabels(labels)
        ax.set_xlabel("")

        for x, y in zip(x_positions[valid_mask], f1_values[valid_mask]):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        return fig

    @render.plot
    def plot_confusion_matrix():
        df_filtered = _filtered_predictions_df()
        fig, ax = plt.subplots(figsize=(3.8, 3.2))

        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "혼동행렬을 표시할 데이터가 없습니다.", ha="center", va="center", fontsize=11)
            return fig

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "혼동행렬을 표시할 데이터가 없습니다.", ha="center", va="center", fontsize=11)
            return fig

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        im = ax.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=12)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["불량", "정상"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["불량", "정상"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.18)
        return fig

    @render.data_frame
    def metrics_table():
        df_filtered = _filtered_predictions_df()
        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            return render.DataGrid(pd.DataFrame({"지표": ["데이터 없음"], "값": ["-"]}))

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            return render.DataGrid(pd.DataFrame({"지표": ["데이터 없음"], "값": ["-"]}))

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)

        metrics = {
            "정확도 (Accuracy)": accuracy_score(y_true, y_pred),
            "정밀도 (Precision)": precision_score(y_true, y_pred, zero_division=0),
            "재현율 (Recall)": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }

        metrics_df = pd.DataFrame({"지표": list(metrics.keys()), "값": [f"{value:.3f}" for value in metrics.values()]})
        return render.DataGrid(metrics_df)

    @render.plot
    def distribution_plot():
        if PI_TOP_FEATURES.empty or TRAIN_DATASET.empty:
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "분포를 표시할 데이터가 없습니다.", ha="center", va="center", fontsize=11)
            return fig

        feature_names = [
            str(name)
            for name in PI_TOP_FEATURES["feature"].astype(str).tolist()
            if name in TRAIN_DATASET.columns
        ][:5]

        if not feature_names:
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "표시할 특징이 없습니다.", ha="center", va="center", fontsize=11)
            return fig

        rows = len(feature_names)
        fig_height = 9.5 * rows if rows > 1 else 10.2
        fig, axes = plt.subplots(rows, 1, figsize=(10.2, fig_height), squeeze=False)
        axes = axes.flatten()

        highlight_rows = _highlighted_prediction_rows()
        non_highlight_rows = _non_highlighted_prediction_rows()
        try:
            selected = input.distribution_datasets()
        except Exception:
            selected = None
        selected_set = set(selected or ["train", "highlight", "non_highlight"])
        if not selected_set:
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha="center", va="center", fontsize=11)
            return fig

        feature_names = [
            str(name)
            for name in PI_TOP_FEATURES["feature"].astype(str).tolist()
            if name in TRAIN_DATASET.columns
        ][:5]

        rows = len(feature_names)
        fig_height = 9.5 * rows if rows > 1 else 10.2
        fig, axes = plt.subplots(rows, 1, figsize=(10.2, fig_height), squeeze=False)
        axes = axes.flatten()

        highlight_rows = _highlighted_prediction_rows()
        non_highlight_rows = _non_highlighted_prediction_rows()
        highlight_rows = highlight_rows.reindex(columns=TRAIN_DATASET.columns, fill_value=pd.NA)
        non_highlight_rows = non_highlight_rows.reindex(columns=TRAIN_DATASET.columns, fill_value=pd.NA)

        for idx, (ax, feature) in enumerate(zip(axes, feature_names), start=1):
            series = TRAIN_DATASET[feature].dropna()
            if series.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10)
                continue

            highlight_series = (
                highlight_rows[feature].dropna()
                if feature in highlight_rows.columns
                else pd.Series(dtype=series.dtype)
            )
            non_highlight_series = (
                non_highlight_rows[feature].dropna()
                if feature in non_highlight_rows.columns
                else pd.Series(dtype=series.dtype)
            )

            plotted = False
            peak = 0.0

            if pd.api.types.is_numeric_dtype(series):
                data_entries: list[dict[str, object]] = []

                if "train" in selected_set:
                    baseline_values = pd.to_numeric(series, errors="coerce").dropna()
                    baseline_values = _apply_feature_filters(baseline_values, feature)
                    if not baseline_values.empty:
                        data_entries.append({
                            "key": "train",
                            "values": baseline_values,
                            "color": "#2f3e46",
                            "alpha": 0.55,
                            "label": "학습 데이터",
                            "line_color": "#2f3e46",
                            "line_label": "학습 평균",
                        })

                if "highlight" in selected_set:
                    highlight_values = pd.to_numeric(highlight_series, errors="coerce").dropna()
                    highlight_values = _apply_feature_filters(highlight_values, feature)
                    if not highlight_values.empty:
                        data_entries.append({
                            "key": "highlight",
                            "values": highlight_values,
                            "color": "#a41623",
                            "alpha": 0.7,
                            "label": "하이라이트 데이터",
                            "line_color": "#a41623",
                            "line_label": "하이라이트 평균",
                        })

                if "non_highlight" in selected_set:
                    normal_values = pd.to_numeric(non_highlight_series, errors="coerce").dropna()
                    normal_values = _apply_feature_filters(normal_values, feature)
                    if not normal_values.empty:
                        data_entries.append({
                            "key": "non_highlight",
                            "values": normal_values,
                            "color": "#918450",
                            "alpha": 0.7,
                            "label": "일반 데이터",
                            "line_color": "#918450",
                            "line_label": "일반 평균",
                        })

                if not data_entries:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10)
                    continue

                bin_reference = next((entry["values"] for entry in data_entries if len(entry["values"]) > 0), None)
                if bin_reference is None:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10)
                    continue

                bin_reference = pd.Series(bin_reference, dtype=float)
                bins = max(5, min(80, int(np.sqrt(len(bin_reference))) * 4))
                weights = None
                bin_edges = None

                for entry in data_entries:
                    values = pd.Series(entry["values"], dtype=float).dropna()
                    if values.empty:
                        continue
                    weights = np.full(len(values), 100.0 / len(values))
                    if bin_edges is None:
                        counts, bin_edges, _ = ax.hist(
                            values,
                            bins=bins,
                            weights=weights,
                            color=entry["color"],
                            alpha=entry["alpha"],
                            edgecolor="white",
                            label=entry["label"],
                        )
                    else:
                        counts, _, _ = ax.hist(
                            values,
                            bins=bin_edges,
                            weights=weights,
                            color=entry["color"],
                            alpha=entry["alpha"],
                            edgecolor="white",
                            label=entry["label"],
                        )
                    if counts.size:
                        peak = max(peak, float(np.max(counts)))
                    mean_value = float(values.mean())
                    ax.axvline(
                        mean_value,
                        color=entry["line_color"],
                        linestyle="--",
                        linewidth=2,
                        label=entry["line_label"],
                    )
                    plotted = True

                ax.set_ylabel("비율 (%)")

            else:
                dataset_counts: list[tuple[str, pd.Series, str]] = []

                if "train" in selected_set:
                    baseline_counts = series.astype(str).value_counts(normalize=True) * 100.0
                    if not baseline_counts.empty:
                        dataset_counts.append(("학습 데이터", baseline_counts, "#2f3e46"))

                if "highlight" in selected_set:
                    highlight_counts = highlight_series.astype(str).value_counts(normalize=True) * 100.0
                    if not highlight_counts.empty:
                        dataset_counts.append(("하이라이트 데이터", highlight_counts, "#a41623"))

                if "non_highlight" in selected_set:
                    normal_counts = non_highlight_series.astype(str).value_counts(normalize=True) * 100.0
                    if not normal_counts.empty:
                        dataset_counts.append(("일반 데이터", normal_counts, "#918450"))

                if not dataset_counts:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10)
                    continue

                order_source = dataset_counts[0][1].sort_values(ascending=False)
                top_categories = order_source.head(20).index.tolist()
                for _, counts, _ in dataset_counts[1:]:
                    for cat in counts.sort_values(ascending=False).index:
                        if cat not in top_categories:
                            top_categories.append(cat)
                        if len(top_categories) >= 20:
                            break
                    if len(top_categories) >= 20:
                        break

                base_vals_list = []
                for label, counts, color in dataset_counts:
                    aligned = counts.reindex(top_categories, fill_value=0.0).astype(float)
                    base_vals_list.append((label, aligned, color))
                    if aligned.sum() > 0:
                        plotted = True
                    peak = max(peak, float(aligned.max()))

                x = np.arange(len(top_categories))
                width = 0.24 if len(base_vals_list) == 3 else 0.32 if len(base_vals_list) == 2 else 0.38
                offset_start = -width * (len(base_vals_list) - 1) / 2

                for idx_shift, (label, aligned, color) in enumerate(base_vals_list):
                    shift = offset_start + idx_shift * width
                    ax.bar(
                        x + shift,
                        aligned.values,
                        width=width,
                        color=color,
                        alpha=0.75 if label == "학습 데이터" else 0.7,
                        label=label,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(top_categories, rotation=30, ha="right")
                ax.set_ylabel("비율 (%)")

            peak = max(5.0, peak)
            ax.set_ylim(0, peak * 1.2)
            ax.margins(x=0.02, y=0.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.yaxis.set_minor_locator(MaxNLocator(nbins=16, integer=True))
            ax.tick_params(axis="y", which="major", length=6)
            ax.tick_params(axis="y", which="minor", length=3)

            if plotted:
                ax.legend(loc="upper right")

            ax.set_title(f"{idx}. {feature}", fontsize=11, loc="left")
            ax.grid(alpha=0.25)

        for ax in axes[len(feature_names):]:
            ax.set_axis_off()

        if len(feature_names) > 1:
            fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.97, hspace=1.6)
        else:
            fig.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.97)
        return fig

    @reactive.effect
    @reactive.event(input.btn_distribution_view)
    def _show_distribution_modal():
        if PI_TOP_FEATURES.empty or TRAIN_DATASET.empty:
            body = ui.p("표시할 데이터가 없습니다.", class_="text-muted")
        else:
            body = ui.div(
                ui.input_checkbox_group(
                    "distribution_datasets",
                    None,
                    choices={
                        "train": "학습 데이터",
                        "highlight": "하이라이트 데이터",
                        "non_highlight": "일반 데이터",
                    },
                    selected=["train", "highlight"],
                    inline=True,
                ),
                ui.output_plot("distribution_plot", height="1200px"),
            )
        ui.modal_show(
            ui.modal(
                ui.card(
                    ui.card_header("변수 중요도 Top 5 학습데이터와의 비교"),
                    body,
                ),
                title="",
                easy_close=True,
                footer=ui.modal_button("닫기"),
                size="xl",
            )
        )

    @render.data_frame
    def modal_low_f1_table():
        df = _highlighted_dates()
        if df.empty:
            return render.DataGrid(df, selection_mode="row")
        return render.DataGrid(df, selection_mode="row")

    @reactive.effect
    def _sync_modal_selection():
        try:
            selected = input.modal_low_f1_table_selected_rows()
        except Exception:
            selected = None
        df = _highlighted_dates()
        if selected and len(selected) > 0 and not df.empty:
            idx = selected[0]
            if 0 <= idx < len(df):
                modal_selected_day.set(df.iloc[idx]["날짜"])
        elif df.empty:
            modal_selected_day.set(None)

    @render.plot
    def modal_confusion_matrix():
        df_filtered, selected_value = _filtered_by_modal_day()
        fig, ax = plt.subplots(figsize=(6.0, 3.8))

        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            ax.set_axis_off()
            msg = "임계선 이하 날짜가 없습니다." if not selected_value else "선택한 날짜에 데이터가 없습니다."
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
            return fig

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            ax.set_axis_off()
            msg = "임계선 이하 날짜가 없습니다." if not selected_value else "선택한 날짜에 데이터가 없습니다."
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
            return fig

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        im = ax.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=12)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["불량", "정상"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["불량", "정상"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.18)
        return fig

    @render.table
    def modal_metrics_table():
        df_filtered, selected_value = _filtered_by_modal_day()
        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            return pd.DataFrame({"지표": ["데이터 없음"], "값": ["-"]})

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            msg = "임계선 이하 날짜가 없습니다." if not selected_value else "선택한 날짜에 데이터가 없습니다."
            return pd.DataFrame({"지표": ["데이터 없음"], "값": [msg]})

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)

        metrics = {
            "정확도 (Accuracy)": accuracy_score(y_true, y_pred),
            "정밀도 (Precision)": precision_score(y_true, y_pred, zero_division=0),
            "재현율 (Recall)": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }

        return pd.DataFrame({"지표": list(metrics.keys()), "값": [f"{value:.3f}" for value in metrics.values()]})

def _load_model_score_f1_values() -> pd.Series:
    try:
        df_scores = pd.read_csv(MODEL_SCORE_FILE)
    except Exception:
        return pd.Series(dtype=float)

    candidate_cols = ("f1_score", "f1", "F1", "f1_score_mean", "f1_mean")
    for col in candidate_cols:
        if col in df_scores.columns:
            values = pd.to_numeric(df_scores[col], errors="coerce").dropna()
            if not values.empty:
                return values

    numeric_cols = df_scores.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        return pd.to_numeric(df_scores[numeric_cols[0]], errors="coerce").dropna()

    return pd.Series(dtype=float)

def _get_model_f1_score() -> float | None:
    return _MODEL_F1_SCORE

_MODEL_F1_VALUES = _load_model_score_f1_values()
_MODEL_F1_SCORE = float(_MODEL_F1_VALUES.mean()) if not _MODEL_F1_VALUES.empty else None
if _MODEL_F1_SCORE is not None:
    THRESHOLD = min(THRESHOLD, float(np.clip(_MODEL_F1_SCORE, 0.0, 1.0)))
