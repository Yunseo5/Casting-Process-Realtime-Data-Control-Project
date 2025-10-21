from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = BASE_DIR / "data" / "intermin" / "test_predictions_v2.csv"
SEGMENT_COUNT = 15
X_AXIS_MARGIN = 1 / 3
MOLD_CODES = ["8412", "8917", "8722", "8413", "8576"]


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


def _segment_f1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"segment": np.arange(1, SEGMENT_COUNT + 1), "timestamp": pd.NaT, "f1": np.nan})

    if "passorfail" not in df.columns or "prediction" not in df.columns:
        raise KeyError("예측 파일에 'passorfail' 또는 'prediction' 컬럼이 없습니다.")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    valid_mask = timestamps.notna()
    valid_df = df.loc[valid_mask].copy()
    valid_times = timestamps.loc[valid_mask]

    segments: list[dict[str, object]] = []

    if not valid_df.empty and valid_times.min() != valid_times.max():
        boundaries = pd.date_range(valid_times.min(), valid_times.max(), periods=SEGMENT_COUNT + 1)
        for idx in range(SEGMENT_COUNT):
            left = boundaries[idx]
            right = boundaries[idx + 1]
            if idx < SEGMENT_COUNT - 1:
                mask = (valid_times >= left) & (valid_times < right)
            else:
                mask = (valid_times >= left) & (valid_times <= right)
            subset = valid_df.loc[mask]
            midpoint = left + (right - left) / 2
            if subset.empty:
                segments.append({"segment": idx + 1, "timestamp": midpoint, "f1": np.nan})
            else:
                f1 = f1_score(subset["passorfail"].astype(int), subset["prediction"].astype(int), zero_division=0)
                segments.append({"segment": idx + 1, "timestamp": midpoint, "f1": f1})
    else:
        indices = np.array_split(df.index.to_numpy(), SEGMENT_COUNT)
        for idx, idx_block in enumerate(indices, start=1):
            if len(idx_block) == 0:
                segments.append({"segment": idx, "timestamp": pd.NaT, "f1": np.nan})
                continue
            slice_df = df.loc[idx_block]
            f1 = f1_score(slice_df["passorfail"].astype(int), slice_df["prediction"].astype(int), zero_division=0)
            ts_subset = timestamps.loc[idx_block]
            midpoint = ts_subset.dropna().iloc[len(ts_subset.dropna()) // 2] if ts_subset.notna().any() else pd.NaT
            segments.append({"segment": idx, "timestamp": midpoint, "f1": f1})

    return pd.DataFrame(segments)


plot_card = ui.card(
    ui.card_header("금형 코드별 F1-Score 추이"),
    ui.output_plot("plot_segment_f1", height="420px"),
    ui.layout_columns(
        ui.output_plot("plot_confusion_matrix", height="260px"),
        ui.div(
            ui.h6("성능 지표", class_="fw-semibold"),
            ui.output_table("metrics_table"),
            class_="mt-2",
        ),
        col_widths=[6, 6],
    ),
)

buttons_card = ui.card(
    ui.card_header("금형 코드"),
    ui.output_ui("mold_buttons"),
    style="flex:2 1 0;width:100%;",
)

distribution_card = ui.card(
    ui.card_header("분포 분석"),
    ui.div(
        ui.input_action_button("btn_distribution_visual", "시각화", class_="btn btn-outline-secondary w-100"),
        ui.input_action_button("btn_distribution_numeric", "수치화", class_="btn btn-outline-secondary w-100"),
        ui.input_action_button("btn_distribution_hypothesis", "가설검정", class_="btn btn-outline-secondary w-100"),
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
        col_widths=[8, 4],
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
            return df_predictions.loc[df_predictions["mold_code"].astype(str) == filter_code]
        return df_predictions

    @render.plot
    def plot_segment_f1():
        df_filtered = _filtered_predictions_df()
        segment_df = _segment_f1(df_filtered)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))

        if segment_df.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha="center", va="center", fontsize=12)
            return fig

        x_positions = np.arange(1, SEGMENT_COUNT + 1)
        f1_values = segment_df["f1"].to_numpy()
        valid_mask = (~np.isnan(f1_values)) & (f1_values > 0)

        threshold = 0.85
        line_color = "#383636"
        above_color = "#383636"
        below_color = "#FF0000"
        highlight_color = "#F9C5C5"

        above_mask = valid_mask & (f1_values >= threshold)
        below_mask = valid_mask & (f1_values < threshold)

        for idx, is_low in enumerate(below_mask, start=1):
            if is_low:
                left = idx - 0.5
                right = idx + 0.5
                ax.axvspan(left, right, color=highlight_color, alpha=0.3, zorder=0)

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
        ax.axhline(threshold, color="#C1121F", linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("")
        ax.grid(alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xlim(0.5 - X_AXIS_MARGIN, SEGMENT_COUNT + 0.5 + X_AXIS_MARGIN)

        if segment_df["timestamp"].notna().any():
            tick_labels = [ts.strftime("%Y-%m-%d") if pd.notna(ts) else "" for ts in segment_df["timestamp"]]
            ax.set_xticklabels(tick_labels, rotation=30, ha="right")
            ax.set_xlabel("")
        else:
            ax.set_xticklabels([str(idx) for idx in x_positions])
            ax.set_xlabel("구간 (1~15)")

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
        fig.tight_layout()
        return fig

    @render.table
    def metrics_table():
        df_filtered = _filtered_predictions_df()
        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            return pd.DataFrame({"지표": ["데이터 없음"], "값": ["-"]})

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            return pd.DataFrame({"지표": ["데이터 없음"], "값": ["-"]})

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)

        metrics = {
            "정확도 (Accuracy)": accuracy_score(y_true, y_pred),
            "정밀도 (Precision)": precision_score(y_true, y_pred, zero_division=0),
            "재현율 (Recall)": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }

        metrics_df = pd.DataFrame(
            {"지표": list(metrics.keys()), "값": [f"{value:.3f}" for value in metrics.values()]}
        )
        return metrics_df

