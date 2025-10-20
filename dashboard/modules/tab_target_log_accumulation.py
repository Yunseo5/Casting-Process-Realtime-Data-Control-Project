from shiny import ui, render, reactive
import pandas as pd

tab_ui = ui.page_fluid(
    ui.div(
        # 헤더
        ui.div(
            ui.h2(
                ui.HTML('<i class="fa-solid fa-database"></i>'),
                "로그 누적",
            ),
            ui.p("전체 누적 데이터 모니터링"),
            class_="tab-header",
        ),
        
        ui.div(class_="divider"),
        
        # 제어 버튼
        ui.div(
            ui.div(
                ui.input_action_button(
                    "tab_log_start_btn", 
                    ui.HTML('<i class="fa-solid fa-play"></i> 시작'),
                    class_="btn-enhanced btn-start w-100"
                ),
                class_="col-12 col-md-4",
            ),
            ui.div(
                ui.input_action_button(
                    "tab_log_stop_btn", 
                    ui.HTML('<i class="fa-solid fa-pause"></i> 정지'),
                    class_="btn-enhanced btn-stop w-100"
                ),
                class_="col-12 col-md-4",
            ),
            ui.div(
                ui.input_action_button(
                    "tab_log_reset_btn", 
                    ui.HTML('<i class="fa-solid fa-rotate-right"></i> 리셋'),
                    class_="btn-enhanced btn-reset w-100"
                ),
                class_="col-12 col-md-4",
            ),
            class_="row control-buttons",
        ),
        
        # 진행 상태 및 통계
        ui.div(
            ui.output_ui("tab_log_progress_text"),
            class_="progress-card",
        ),
        
        # 통계 정보
        ui.div(
            ui.output_ui("tab_log_stats"),
            class_="stats-card",
        ),
        
        # DataGrid (전체 데이터)
        ui.div(
            ui.div(
                ui.HTML('<i class="fa-solid fa-table-list"></i>'),
                "누적 데이터 (전체)",
                class_="datagrid-header",
            ),
            ui.output_data_frame("tab_log_table_all"),
            class_="datagrid-container",
        ),
        
        class_="main-content-card tab-log",  # ✅ tab-log 클래스 (보라색 테마)
    ),
)

def tab_server(input, output, session, streamer, shared_df, streaming_active):

    @reactive.effect
    @reactive.event(input.tab_log_start_btn)
    def _on_start():
        streamer.start_stream()
        streaming_active.set(True)

    @reactive.effect
    @reactive.event(input.tab_log_stop_btn)
    def _on_stop():
        streamer.stop_stream()
        streaming_active.set(False)

    @reactive.effect
    @reactive.event(input.tab_log_reset_btn)
    def _on_reset():
        streamer.reset_stream()
        shared_df.set(pd.DataFrame())
        streaming_active.set(False)

    @output
    @render.ui
    def tab_log_progress_text():
        status = streaming_active.get()
        _ = shared_df.get()
        
        if status:
            icon = '<i class="fa-solid fa-circle-play"></i>'
            status_text = "진행 중"
        else:
            icon = '<i class="fa-solid fa-circle-pause"></i>'
            status_text = "정지"
        
        progress = streamer.progress()
        
        return ui.div(
            ui.HTML(
                f'{icon} 상태: {status_text} | 진행률: {progress:.1f}%'
            ),
            class_="progress-text",
        )

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
                style="font-weight: 600; font-size: 16px; color: #2c3e50;",
            ),
            ui.div(
                ui.HTML('<i class="fa-solid fa-memory"></i>'),
                f" 메모리 사용량: {memory_usage}",
                style="font-weight: 600; font-size: 16px; color: #2c3e50; margin-top: 10px;",
            ),
        )

    @output
    @render.data_frame
    def tab_log_table_all():
        df = shared_df.get()
        
        # 데이터가 없을 때
        if df.empty:
            empty_df = pd.DataFrame({
                "메시지": ["⏳ 데이터를 불러오는 중..."]
            })
            return render.DataGrid(
                empty_df,
                height="600px",
                width="100%",
                filters=True,
                row_selection_mode="none",
            )
        
        # 전체 데이터 표시 (불필요한 컬럼 제거)
        result = df.drop(columns=['line', 'name', 'mold_name'], errors='ignore').copy()
        
        return render.DataGrid(
            result,
            height="600px",
            width="100%",
            filters=True,
            row_selection_mode="none",
        )