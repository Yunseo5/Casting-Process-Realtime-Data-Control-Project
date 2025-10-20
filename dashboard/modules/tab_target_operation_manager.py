from shiny import ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
def setup_korean_font():
    fonts = [f.name for f in fm.fontManager.ttflist]
    for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans KR']:
        if font in fonts:
            plt.rcParams['font.family'] = font
            return
    plt.rcParams['font.family'] = 'DejaVu Sans'

setup_korean_font()
plt.rcParams['axes.unicode_minus'] = False

# 상수
MOLD_CODES = ["8412", "8917", "8722", "8413", "8576"]
VARIABLES = {
    "molten_temp": "용탕 온도 (molten_temp)",
    "facility_operation_cycleTime": "설비 사이클 타임 (facility_operation_cycleTime)",
    "production_cycletime": "생산 사이클 타임 (production_cycletime)",
    "low_section_speed": "저속 구간 속도 (low_section_speed)",
    "high_section_speed": "고속 구간 속도 (high_section_speed)",
    "molten_volume": "용탕 부피 (molten_volume)",
    "cast_pressure": "주조 압력 (cast_pressure)",
    "biscuit_thickness": "비스킷 두께 (biscuit_thickness)",
    "upper_mold_temp1": "상부 금형 온도1 (upper_mold_temp1)",
    "upper_mold_temp2": "상부 금형 온도2 (upper_mold_temp2)",
    "upper_mold_temp3": "상부 금형 온도3 (upper_mold_temp3)",
    "lower_mold_temp1": "하부 금형 온도1 (lower_mold_temp1)",
    "lower_mold_temp2": "하부 금형 온도2 (lower_mold_temp2)",
    "lower_mold_temp3": "하부 금형 온도3 (lower_mold_temp3)",
    "sleeve_temperature": "슬리브 온도 (sleeve_temperature)",
    "physical_strength": "물리적 강도 (physical_strength)",
    "Coolant_temperature": "냉각수 온도 (Coolant_temperature)",
    "EMS_operation_time": "EMS 작동 시간 (EMS_operation_time)"
}

# CSS - 스크롤 안정 + 여백 최소화
STYLES = """
.main-container{max-width:1400px;margin:0 auto;padding:20px 0}
.kpi-card{background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:30px 20px;text-align:center;
box-shadow:0 2px 4px rgba(0,0,0,.05);transition:all .2s;height:200px;min-height:200px;max-height:200px;
display:flex;flex-direction:column;justify-content:center;overflow:hidden}
.kpi-card:hover{transform:translateY(-4px);box-shadow:0 6px 16px rgba(0,0,0,.15)}
.kpi-title{font-size:15px;font-weight:600;color:#333;margin-bottom:10px}
.kpi-line{width:60%;height:3px;margin:0 auto 15px;border-radius:2px}
.red-line{background:#d9534f}.yellow-line{background:#f0ad4e}.navy-line{background:#2c3e50}
.kpi-value{font-size:28px;font-weight:700;color:#111;margin-bottom:8px}
.kpi-sub{font-size:13px;color:#777}
.status-panel{display:flex;flex-direction:column;justify-content:center;align-items:flex-start;
height:200px;min-height:200px;max-height:200px;gap:18px;padding:20px 30px;background:#fff;
border:1px solid #e0e0e0;border-radius:12px;box-shadow:0 2px 4px rgba(0,0,0,.05);min-width:280px;overflow:hidden}
.status-row{display:flex;align-items:center;gap:16px;width:100%}
.status-indicator-circle{width:50px;height:50px;border-radius:50%;border:3px solid #333;flex-shrink:0}
#lof_indicator{background:#C00000;box-shadow:0 0 22px rgba(192,0,0,.55)}
#process_indicator{background:#3B7D23;box-shadow:0 0 20px rgba(59,125,35,.5)}
#defect_indicator{background:#3B7D23;box-shadow:0 0 20px rgba(59,125,35,.5)}
.status-indicator-label{font-size:15px;font-weight:700;color:#111;white-space:nowrap}
.control-section{background:#fff;border-radius:16px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.08);
padding:24px 20px 36px 20px;height:170px;min-height:170px;max-height:170px;overflow:hidden}
.control-buttons-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:22px;
height:56px;min-height:56px;max-height:56px}
.btn-enhanced{height:56px;border-radius:14px;font-weight:600;font-size:16px;border:none;display:flex;
align-items:center;justify-content:center;gap:10px;transition:all .3s;box-shadow:0 4px 12px rgba(0,0,0,.1);white-space:nowrap}
.btn-enhanced:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.15)}
.btn-start{background:linear-gradient(135deg,#27ae60 0%,#2ecc71 100%);color:#fff}
.btn-stop{background:linear-gradient(135deg,#f39c12 0%,#f1c40f 100%);color:#fff}
.btn-reset{background:linear-gradient(135deg,#e74c3c 0%,#c0392b 100%);color:#fff}
.progress-bar{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:12px;padding:0 24px;
color:#fff;font-size:16px;font-weight:600;display:flex;align-items:center;justify-content:center;
height:56px;min-height:56px;max-height:56px;width:100%;overflow:hidden}
.progress-bar>div{width:100%;text-align:center;white-space:nowrap}
.custom-card{background:#fff;border-radius:16px;padding:24px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card-header-title{font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;
border-bottom:2px solid #e0e0e0;height:40px;min-height:40px;max-height:40px;overflow:hidden}
.chart-container{height:450px;min-height:450px;max-height:450px;width:100%;position:relative;overflow:hidden}
.table-container{width:100%;overflow-x:auto;overflow-y:auto;border:1px solid #e0e0e0;border-radius:8px;
height:480px;min-height:480px;max-height:480px}
.table-container::-webkit-scrollbar{height:12px;width:12px}
.table-container::-webkit-scrollbar-track{background:#f1f1f1;border-radius:6px}
.table-container::-webkit-scrollbar-thumb{background:#888;border-radius:6px}
.table-container::-webkit-scrollbar-thumb:hover{background:#555}
.shiny-data-grid{font-size:14px!important;width:100%;border-collapse:collapse}
.shiny-data-grid thead{background:linear-gradient(135deg,#4A90E2 0%,#357ABD 100%);position:sticky;top:0;z-index:10}
.shiny-data-grid thead th{font-weight:600;padding:14px 12px;color:#2A2D30;white-space:nowrap}
.shiny-data-grid tbody tr:hover{background:#f8f9fa}
.shiny-data-grid tbody td{padding:12px;border-bottom:1px solid #ecf0f1;white-space:nowrap}
"""

# 헬퍼 함수
def create_kpi(title, output_id, subtitle, line_class):
    return ui.div(ui.p(title, class_="kpi-title"), ui.div(class_=f"kpi-line {line_class}"),
                  ui.output_ui(output_id), ui.p(subtitle, class_="kpi-sub"), class_="kpi-card")

def create_light(light_id, label):
    return ui.div(ui.div(id=light_id, class_="status-indicator-circle"),
                  ui.div(label, class_="status-indicator-label"), class_="status-row")

def get_label(variable):
    return VARIABLES.get(variable, variable).split(' (')[0]

# UI
tab_ui = ui.page_fluid(
    ui.tags.style(STYLES),
    ui.div(
        ui.div(
            ui.div(ui.layout_columns(
                create_kpi("금형별 수율", "kpi_yield", "실시간 집계", "red-line"),
                create_kpi("제품 사이클 타임", "kpi_cycle", "평균 사이클 타임", "yellow-line"),
                create_kpi("설비 가동률", "kpi_uptime", "현재 가동 상태", "navy-line"),
                col_widths=[4,4,4]), style="flex:3"),
            ui.div(ui.div(create_light("lof_indicator", "이상치 발생"),
                         create_light("process_indicator", "관리도 이상"),
                         create_light("defect_indicator", "불량 발생"),
                         class_="status-panel"), style="flex:1"),
            style="display:flex;gap:24px;margin-bottom:24px"),
        ui.div(
            ui.div(ui.div(ui.input_action_button("tab1_start_btn", ui.HTML('<i class="fa-solid fa-play"></i> 시작'), class_="btn-enhanced btn-start")),
                   ui.div(ui.input_action_button("tab1_stop_btn", ui.HTML('<i class="fa-solid fa-pause"></i> 정지'), class_="btn-enhanced btn-stop")),
                   ui.div(ui.input_action_button("tab1_reset_btn", ui.HTML('<i class="fa-solid fa-rotate-right"></i> 리셋'), class_="btn-enhanced btn-reset")),
                   class_="control-buttons-grid"),
            ui.div(ui.output_ui("tab1_progress_text"), class_="progress-bar"),
            class_="control-section"),
        ui.div(ui.div("검색 및 설정", class_="card-header-title"),
               ui.layout_columns(
                   ui.input_selectize("mold_code_select", "Mold Code 검색", choices=MOLD_CODES, selected=None, multiple=False),
                   ui.input_selectize("variable_select", "변수 설정", choices=VARIABLES, selected=None, multiple=False),
                   col_widths=[6,6]), class_="custom-card"),
        ui.div(ui.div("런 차트", class_="card-header-title"),
               ui.div(ui.output_plot("run_chart", height="450px"), class_="chart-container"),
               class_="custom-card"),
        ui.div(ui.div("Top 10 로그 (실시간 데이터)", class_="card-header-title"),
               ui.div(ui.output_data_frame("tab1_table_realtime"), class_="table-container"),
               class_="custom-card"),
        class_="main-container"),
)

# SERVER
def tab_server(input, output, session, streamer, shared_df, streaming_active):
    
    @reactive.effect
    @reactive.event(input.tab1_start_btn)
    def _start():
        streamer.start_stream()
        streaming_active.set(True)

    @reactive.effect
    @reactive.event(input.tab1_stop_btn)
    def _stop():
        streamer.stop_stream()
        streaming_active.set(False)

    @reactive.effect
    @reactive.event(input.tab1_reset_btn)
    def _reset():
        streamer.reset_stream()
        shared_df.set(pd.DataFrame())
        streaming_active.set(False)

    @output
    @render.ui
    def kpi_yield():
        return ui.h1("95.5%", class_="kpi-value")

    @output
    @render.ui
    def kpi_cycle():
        df = shared_df.get()
        if df.empty or 'production_cycletime' not in df.columns:
            return ui.h1("0.0 sec", class_="kpi-value")
        return ui.h1(f"{df['production_cycletime'].mean():.1f} sec", class_="kpi-value")

    @output
    @render.ui
    def kpi_uptime():
        df = shared_df.get()
        if df.empty or 'working' not in df.columns:
            return ui.h1("0.0%", class_="kpi-value")
        working = (df['working'] == '가동').sum()
        return ui.h1(f"{(working/len(df)*100):.1f}%", class_="kpi-value")

    @output
    @render.ui
    def tab1_progress_text():
        _ = shared_df.get()
        status = "진행 중" if streaming_active.get() else "정지"
        return ui.div(f'상태: {status} | 진행률: {streamer.progress():.1f}%')

    @output
    @render.plot
    def run_chart():
        df = shared_df.get()
        mold_code = input.mold_code_select()
        variable = input.variable_select()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks([])
        
        def empty(title, msg='', ylabel='값', color='black'):
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=color)
            if msg:
                ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='#999', transform=ax.transAxes)
            plt.tight_layout()
            return fig
        
        if not mold_code or not variable:
            return empty('Mold Code와 변수를 선택하세요')
        
        label = get_label(variable)
        
        if df.empty:
            return empty(f'Mold Code {mold_code} - {label}', ylabel=label)
        
        try:
            if 'mold_code' not in df.columns:
                return empty(f'Mold Code {mold_code} - {label}', 'mold_code 컬럼 없음', label, '#e74c3c')
            
            filtered = df[df['mold_code'] == int(mold_code)].copy()
            
            if filtered.empty:
                codes = sorted(df['mold_code'].unique())
                return empty(f'Mold Code {mold_code} - {label}', f'데이터 없음\n존재: {codes}', label, '#e74c3c')
            
            if variable not in filtered.columns:
                return empty(f'Mold Code {mold_code} - {label}', f'"{label}" 없음', label, '#e74c3c')
            
            if 'registration_time' in filtered.columns:
                filtered = filtered.sort_values('registration_time')
            
            plot_df = filtered[[variable]].dropna().reset_index(drop=True).tail(10)
            
            if plot_df.empty:
                return empty(f'Mold Code {mold_code} - {label}', '결측치만 존재', label, '#e74c3c')
            
            x, y = range(len(plot_df)), plot_df[variable].values
            
            ax.plot(x, y, color='#4A90E2', linewidth=2, marker='o', markersize=6,
                   markerfacecolor='#2c3e50', markeredgecolor='white', markeredgewidth=1.5)
            
            mean = y.mean()
            ax.axhline(y=mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'평균: {mean:.2f}')
            
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'Mold Code {mold_code} - {label} ({len(plot_df)}개 데이터)', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return empty('오류 발생', str(e), color='#e74c3c')

    @output
    @render.data_frame
    def tab1_table_realtime():
        df = shared_df.get()
        
        if df.empty:
            return render.DataGrid(
                pd.DataFrame({"메시지": ["데이터를 불러오는 중..."] + [""] * 9}),
                width="100%", filters=False, row_selection_mode="none"
            )
        
        result = df.tail(10).drop(columns=['line', 'name', 'mold_name'], errors='ignore').copy()
        
        if len(result) < 10:
            pad = pd.DataFrame([[None] * len(result.columns)] * (10 - len(result)), columns=result.columns)
            result = pd.concat([result, pad], ignore_index=True)
        
        return render.DataGrid(result, width="100%", filters=False, row_selection_mode="none")