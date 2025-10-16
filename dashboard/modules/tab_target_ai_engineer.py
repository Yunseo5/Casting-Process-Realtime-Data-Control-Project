# dashboard/modules/tab_target_ai_engineer.py
from shiny import ui, render
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 샘플 데이터 생성 --------------------------------------------------------------
_rng = np.random.default_rng(2024)

EDA_DATA = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=120, freq="H"),
        "temperature": _rng.normal(loc=720, scale=12, size=120),
        "pressure": _rng.normal(loc=45, scale=4, size=120),
        "speed": _rng.normal(loc=32, scale=3, size=120),
    }
)
EDA_DATA["defect"] = np.where(
    (_rng.random(120) > 0.82) | (EDA_DATA["temperature"] > 735), "결함", "정상"
)

MODEL_CUMULATIVE = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "value": np.cumsum(_rng.normal(loc=48, scale=5, size=30)),
    }
)

MODEL_CONFUSION = np.array([[54, 7], [5, 34]])

PDP_VALUES = np.linspace(0, 1, 30)
PDP_A = 0.6 + 0.25 * np.sin(PDP_VALUES * np.pi)
PDP_B = 0.5 + 0.3 * np.cos(PDP_VALUES * np.pi)
PDP_C = 0.55 + 0.2 * np.sin(PDP_VALUES * np.pi * 1.5)

FEATURE_IMPORTANCE = pd.Series(
    [0.32, 0.26, 0.18, 0.12, 0.07, 0.05],
    index=["temperature", "pressure", "speed", "humidity", "alloy_mix", "coolant"],
)

SHAP_SUMMARY = pd.DataFrame(
    {
        "feature": np.repeat(["temperature", "pressure", "speed"], repeats=50),
        "value": np.concatenate(
            [
                _rng.normal(0.45, 0.12, 50),
                _rng.normal(-0.1, 0.08, 50),
                _rng.normal(0.2, 0.1, 50),
            ]
        ),
        "impact": np.concatenate(
            [
                _rng.normal(0.18, 0.04, 50),
                _rng.normal(0.07, 0.03, 50),
                _rng.normal(0.1, 0.02, 50),
            ]
        ),
    }
)


# 탭별 UI ----------------------------------------------------------------------
tab_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel(
            "EDA",
            ui.layout_columns(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("온도 분포"),
                        ui.output_plot("plot_feature_distribution", height="300px"),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("상관 행렬"),
                        ui.output_plot("plot_correlation_matrix", height="300px"),
                    ),
                ),
            ),
            ui.layout_columns(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("최근 샘플"),
                        ui.output_table("table_recent_samples"),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("통계 요약"),
                        ui.output_table("table_stats_summary"),
                    ),
                ),
            ),
        ),
        ui.nav_panel(
            "MODEL",
            ui.layout_columns(
                ui.column(
                    7,
                    ui.card(
                        ui.card_header("혼동행렬"),
                        ui.output_plot("plot_confusion_matrix", height="320px"),
                    ),
                ),
                ui.column(
                    5,
                    ui.card(
                        ui.card_header("누적값 (기간 설정)"),
                        ui.input_date_range(
                            "date_range",
                            "기간",
                            start="2024-01-01",
                            end="2024-01-31",
                        ),
                        ui.output_plot("plot_cumulative", height="260px"),
                    ),
                ),
            ),
            ui.card(
                ui.card_header("변수설정 (PDP)"),
                ui.layout_columns(
                    ui.column(
                        4,
                        ui.card(
                            ui.card_header("변수 A"),
                            ui.output_plot("plot_pdp_a", height="200px"),
                        ),
                    ),
                    ui.column(
                        4,
                        ui.card(
                            ui.card_header("변수 B"),
                            ui.output_plot("plot_pdp_b", height="200px"),
                        ),
                    ),
                    ui.column(
                        4,
                        ui.card(
                            ui.card_header("변수 C"),
                            ui.output_plot("plot_pdp_c", height="200px"),
                        ),
                    ),
                ),
            ),
            ui.layout_columns(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("성능지표"),
                        ui.output_table("table_metrics"),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("이상치"),
                        ui.output_table("table_anomalies"),
                    ),
                ),
            ),
        ),
        ui.nav_panel(
            "해석",
            ui.card(
                ui.card_header("모델 개요"),
                ui.output_text("text_model_overview"),
            ),
            ui.layout_columns(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Feature Importance"),
                        ui.output_plot("plot_feature_importance", height="280px"),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("국소 영향도 (SHAP)"),
                        ui.output_plot("plot_shap_summary", height="280px"),
                    ),
                ),
            ),
            ui.layout_columns(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("해석 요약"),
                        ui.output_table("table_recommendations"),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Action Items"),
                        ui.output_ui("ui_action_items"),
                    ),
                ),
            ),
        ),
    ),
)


# 탭별 서버 --------------------------------------------------------------------
def tab_server(input, output, session):
    @render.plot
    def plot_feature_distribution():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(EDA_DATA["temperature"], bins=20, color="#4C72B0", alpha=0.85)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.plot
    def plot_correlation_matrix():
        corr = EDA_DATA[["temperature", "pressure", "speed"]].corr()
        fig, ax = plt.subplots(figsize=(4, 3))
        cax = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        for (i, j), value in np.ndenumerate(corr.values):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    @render.table
    def table_recent_samples():
        return (
            EDA_DATA.tail(8)
            .loc[:, ["timestamp", "temperature", "pressure", "speed", "defect"]]
            .reset_index(drop=True)
        )

    @render.table
    def table_stats_summary():
        return (
            EDA_DATA[["temperature", "pressure", "speed"]]
            .describe()
            .round(2)
            .rename(index={"50%": "median"})
        )

    @render.plot
    def plot_confusion_matrix():
        fig, ax = plt.subplots(figsize=(4, 3))
        cax = ax.imshow(MODEL_CONFUSION, cmap="Greens")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["정상", "결함"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["정상", "결함"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        for (i, j), value in np.ndenumerate(MODEL_CONFUSION):
            ax.text(j, i, int(value), ha="center", va="center", color="black")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    @render.plot
    def plot_cumulative():
        date_range = MODEL_CUMULATIVE.copy()
        start, end = input.date_range()
        if start is not None and end is not None:
            mask = (date_range["date"] >= pd.to_datetime(start)) & (
                date_range["date"] <= pd.to_datetime(end)
            )
            date_range = date_range.loc[mask]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(date_range["date"], date_range["value"], marker="o", color="#2A9D8F")
        ax.fill_between(
            date_range["date"],
            date_range["value"],
            alpha=0.1,
            color="#2A9D8F",
        )
        ax.set_ylabel("누적 생산량")
        ax.set_xlabel("날짜")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    @render.plot
    def plot_pdp_a():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(PDP_VALUES, PDP_A, color="#003f5c")
        ax.set_title("온도 영향")
        ax.set_xlabel("Normalized Feature")
        ax.set_ylabel("예측 영향")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.plot
    def plot_pdp_b():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(PDP_VALUES, PDP_B, color="#58508d")
        ax.set_title("압력 영향")
        ax.set_xlabel("Normalized Feature")
        ax.set_ylabel("예측 영향")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.plot
    def plot_pdp_c():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(PDP_VALUES, PDP_C, color="#bc5090")
        ax.set_title("속도 영향")
        ax.set_xlabel("Normalized Feature")
        ax.set_ylabel("예측 영향")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.table
    def table_metrics():
        return pd.DataFrame(
            [
                {"지표": "정확도", "값": "0.91"},
                {"지표": "재현율", "값": "0.87"},
                {"지표": "정밀도", "값": "0.89"},
                {"지표": "F1-score", "값": "0.88"},
            ]
        )

    @render.table
    def table_anomalies():
        anomalies = (
            EDA_DATA.loc[EDA_DATA["defect"] == "결함", ["timestamp", "temperature", "pressure"]]
            .head(6)
            .rename(columns={"timestamp": "시간"})
        )
        return anomalies.reset_index(drop=True)

    @render.text
    def text_model_overview():
        return (
            "현재 모델은 Gradient Boosting 기반으로 학습되었으며, "
            "최근 30일 데이터에 대해 약 91% 정확도를 유지하고 있습니다."
        )

    @render.plot
    def plot_feature_importance():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(FEATURE_IMPORTANCE.index[::-1], FEATURE_IMPORTANCE.values[::-1], color="#264653")
        ax.set_xlabel("Importance")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        return fig

    @render.plot
    def plot_shap_summary():
        fig, ax = plt.subplots(figsize=(4, 3))
        for feature, color in zip(
            ["temperature", "pressure", "speed"], ["#1d3557", "#457b9d", "#a8dadc"]
        ):
            subset = SHAP_SUMMARY.loc[SHAP_SUMMARY["feature"] == feature]
            ax.scatter(subset["value"], subset["impact"], alpha=0.6, label=feature, color=color)
        ax.set_xlabel("특징 값 (정규화)")
        ax.set_ylabel("SHAP impact")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @render.table
    def table_recommendations():
        return pd.DataFrame(
            [
                {"구분": "Feature Tuning", "내용": "압력 센서 값이 42~48 구간일 때 품질 안정화"},
                {"구분": "모니터링", "내용": "온도 편차 ±15° 이상 발생 시 경보"},
                {"구분": "추가 분석", "내용": "속도 변수와 냉각 지표 간 교차효과 검증"},
            ]
        )

    @render.ui
    def ui_action_items():
        return ui.tags.ul(
            ui.tags.li("모델 재학습 주기를 월 1회로 고정"),
            ui.tags.li("결함 샘플 확보를 위한 추가 라벨링 진행"),
            ui.tags.li("해석 리포트 자동화를 위한 템플릿 개선"),
        )
