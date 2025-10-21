from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = BASE_DIR / "data" / "intermin" / "test_predictions_v2.csv"
SEGMENT_COUNT = 15
THRESHOLD = 0.85
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


plot_card = ui.card(
    ui.card_header("금형 코드별 F1-Score 데이터"),
    ui.output_data_frame("segment_f1_grid"),
    ui.layout_columns(
        ui.div(
            ui.h6("혼동행렬", class_="fw-semibold"),
            ui.output_data_frame("confusion_matrix_grid"),
        ),
        ui.div(
            ui.h6("성능 지표", class_="fw-semibold"),
            ui.output_data_frame("metrics_grid"),
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
        ui.input_action_button("btn_confusion_detail", "세부 혼동행렬 확인", class_="btn btn-outline-secondary w-100"),
        ui.input_action_button("btn_distribution_view", "분포확인", class_="btn btn-outline-secondary w-100"),
        ui.input_action_button("btn_distribution_pdf_save", "PDF 저장", class_="btn btn-outline-secondary w-100"),
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
            return df_predictions.loc[df_predictions["mold_code"].astype(str) == filter_code]
        return df_predictions

    def _segment_with_flags() -> pd.DataFrame:
        segment_df = _segment_f1(_filtered_predictions_df())
        if segment_df.empty:
            return segment_df.assign(valid=False, below_threshold=False)
        f1_values = segment_df["f1"].to_numpy(dtype=float)
        valid_mask = (~np.isnan(f1_values)) & (f1_values > 0)
        below_mask = valid_mask & (f1_values < THRESHOLD)
        return segment_df.assign(valid=valid_mask, below_threshold=below_mask)

    def _highlighted_dates() -> pd.DataFrame:
        segment_df = _segment_with_flags()
        if segment_df.empty:
            return pd.DataFrame(columns=["날짜", "F1-score"])
        highlight_df = segment_df.loc[segment_df["below_threshold"], ["label", "f1"]].copy()
        highlight_df.columns = ["날짜", "F1-score"]
        highlight_df.sort_values("날짜", inplace=True)
        highlight_df["F1-score"] = highlight_df["F1-score"].astype(float).round(2)
        return highlight_df.reset_index(drop=True)

    def _filtered_by_modal_day() -> tuple[pd.DataFrame, str | None]:
        df_filtered = _filtered_predictions_df()
        selected_value = modal_selected_day.get()
        if selected_value and "timestamp" in df_filtered.columns:
            timestamps = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
            df_filtered = df_filtered.loc[timestamps.dt.strftime("%Y-%m-%d") == selected_value]
        elif not selected_value:
            df_filtered = df_filtered.iloc[0:0]
        return df_filtered, selected_value

    def _highlighted_dates() -> pd.DataFrame:
        segment_df = _segment_with_flags()
        if segment_df.empty:
            return pd.DataFrame(columns=["날짜", "F1-score"])
        highlight_df = segment_df.loc[segment_df["below_threshold"], ["label", "f1"]].copy()
        highlight_df.columns = ["날짜", "F1-score"]
        highlight_df.sort_values("날짜", inplace=True)
        highlight_df["F1-score"] = highlight_df["F1-score"].astype(float).round(2)
        return highlight_df.reset_index(drop=True)

    def _filtered_by_modal_day() -> tuple[pd.DataFrame, str]:
        df_filtered = _filtered_predictions_df()
        selected_value = modal_selected_day.get()
        if selected_value and "timestamp" in df_filtered.columns:
            timestamps = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
            df_filtered = df_filtered.loc[timestamps.dt.strftime("%Y-%m-%d") == selected_value]
        return df_filtered, selected_value

    @render.data_frame
    def segment_f1_grid():
        segment_df = _segment_with_flags()
        if segment_df.empty:
            empty_df = pd.DataFrame({"안내": ["표시할 데이터가 없습니다."]})
            return render.DataGrid(empty_df)

        display_df = segment_df.loc[:, ["label", "f1", "valid", "below_threshold"]].copy()
        display_df.rename(
            columns={
                "label": "구간/날짜",
                "f1": "F1-score",
                "valid": "유효 여부",
                "below_threshold": "임계선 이하",
            },
            inplace=True,
        )
        display_df["F1-score"] = display_df["F1-score"].astype(float).round(3)
        display_df["유효 여부"] = np.where(display_df["유효 여부"], "예", "아니오")
        display_df["임계선 이하"] = np.where(display_df["임계선 이하"], "예", "아니오")
        return render.DataGrid(display_df)

    @render.data_frame
    def confusion_matrix_grid():
        df_filtered = _filtered_predictions_df()
        required_cols = {"passorfail", "prediction"}
        if df_filtered.empty or not required_cols.issubset(df_filtered.columns):
            msg_df = pd.DataFrame({"안내": ["혼동행렬을 표시할 데이터가 없습니다."]})
            return render.DataGrid(msg_df)

        data = df_filtered.loc[:, ["passorfail", "prediction"]].dropna()
        if data.empty:
            msg_df = pd.DataFrame({"안내": ["혼동행렬을 표시할 데이터가 없습니다."]})
            return render.DataGrid(msg_df)

        y_true = data["passorfail"].astype(int)
        y_pred = data["prediction"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        cm_df = pd.DataFrame(
            {
                "실제": ["불량", "정상"],
                "예측 불량": cm[:, 0],
                "예측 정상": cm[:, 1],
            }
        )
        return render.DataGrid(cm_df)

    @render.data_frame
    def metrics_grid():
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

    @reactive.effect
    @reactive.event(input.btn_distribution_view)
    def _show_distribution_modal():
        ui.modal_show(
            ui.modal(
                ui.p(""),
                title="",
                easy_close=True,
                footer=ui.modal_button("닫기"),
                size="l",
            )
        )

    @reactive.effect
    @reactive.event(input.btn_confusion_detail)
    def _show_confusion_detail_modal():
        highlight_df = _highlighted_dates()
        default_day = highlight_df["날짜"].iloc[0] if not highlight_df.empty else None
        modal_selected_day.set(default_day)

        selected_label = selected_mold.get() or '-'

        table_children = [
            ui.card_header("임계선 이하 날짜"),
            ui.p(f"선택된 금형 코드: {selected_label}"),
            ui.output_data_frame("modal_low_f1_table"),
        ]
        if highlight_df.empty:
            table_children.append(ui.p("임계선 이하 날짜가 없습니다.", class_="text-muted"))

        left_card = ui.card(*table_children)

        right_card = ui.card(
            ui.card_header("혼동행렬 및 성능 지표"),
            ui.layout_columns(
                ui.output_plot("modal_confusion_matrix", height="300px"),
                ui.output_table("modal_metrics_table"),
                col_widths=[6, 6],
            ),
        )

        modal_body = ui.layout_columns(
            left_card,
            right_card,
            col_widths=[4, 8],
            class_="align-items-start gap-3",
        )

        ui.modal_show(
            ui.modal(
                modal_body,
                title="",
                easy_close=True,
                footer=ui.modal_button("닫기"),
                size="l",
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
