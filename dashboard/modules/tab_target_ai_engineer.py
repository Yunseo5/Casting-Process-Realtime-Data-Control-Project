# dashboard/modules/tab_target_operation_manager.py
from shiny import ui, render

# 탭별 UI
tab_ui = ui.page_fluid(
    ui.h2("데이터 분석가 탭"),
    ui.output_plot("plot_realtime"),
    ui.output_table("table_kpi")
)

# 탭별 서버
def tab_server(input, output, session):
    @render.plot
    def plot_realtime():
        # 여기에 Run chart 코드 작성
        pass

    @render.table
    def table_kpi():
        # 여기에 mold_code별 KPI 계산 코드 작성
        pass
