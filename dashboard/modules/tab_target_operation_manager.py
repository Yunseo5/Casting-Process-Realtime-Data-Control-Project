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
MOLD_CODES = ["all", "8412", "8917", "8722", "8413", "8576"]
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

# 기본 선택 변수 (처음 3개)
DEFAULT_VARIABLES = ["molten_temp", "facility_operation_cycleTime", "production_cycletime"]

# CSS
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
.status-panel{display:flex;flex-direction:column;justify-content:center;align-items:center;
height:200px;min-height:200px;max-height:200px;padding:20px 30px;background:transparent;
border:none;border-radius:0;box-shadow:none;min-width:280px;overflow:hidden}
.status-row{display:flex;flex-direction:column;align-items:center;gap:12px;width:100%}
.status-indicator-circle{width:70px;height:70px;border-radius:50%;border:4px solid #333;flex-shrink:0;
transition:all 0.3s ease;background:#666;box-shadow:none}
.status-indicator-circle.active-defect{background:#C00000;box-shadow:0 0 30px rgba(192,0,0,.6)}
.status-indicator-label{font-size:17px;font-weight:700;color:#111;white-space:nowrap;text-align:center}
.custom-card{background:#fff;border-radius:16px;padding:24px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card-header-title{font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;
border-bottom:2px solid #e0e0e0}
.chart-container{height:450px;min-height:450px;max-height:450px;width:100%;position:relative;overflow:hidden;margin-bottom:20px}
.table-container{width:100%;overflow-x:auto;overflow-y:auto;border:1px solid #e0e0e0;border-radius:8px;
height:480px;min-height:480px;max-height:480px}
.table-container::-webkit-scrollbar{height:12px;width:12px}
.table-container::-webkit-scrollbar-track{background:#f1f1f1;border-radius:6px}
.table-container::-webkit-scrollbar-thumb{background:#888;border-radius:6px}
.shiny-data-grid{font-size:14px!important;width:100%;border-collapse:collapse}
.shiny-data-grid thead{background:linear-gradient(135deg,#4A90E2 0%,#357ABD 100%);position:sticky;top:0;z-index:10}
.shiny-data-grid thead th{font-weight:600;padding:14px 12px;color:#2A2D30;white-space:nowrap}
.shiny-data-grid tbody tr:hover{background:#f8f9fa}
.shiny-data-grid tbody td{padding:12px;border-bottom:1px solid #ecf0f1;white-space:nowrap}
.toggle-switch{width:40px;height:24px;background:#d0d0d0;border-radius:12px;position:relative;
transition:background 0.3s;border:none;display:block;cursor:pointer}
.toggle-switch.active{background:#2c3e50}
.toggle-circle{position:absolute;width:16px;height:16px;background:white;border-radius:50%;top:4px;left:4px;
transition:left 0.3s;box-shadow:0 2px 4px rgba(0,0,0,0.2)}
.toggle-switch.active .toggle-circle{left:20px}
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
            ui.div(ui.div(create_light("defect_indicator", "불량 발생"),
                         class_="status-panel"), style="flex:1"),
            style="display:flex;gap:24px;margin-bottom:24px"),
        ui.div(ui.div("검색 및 설정", class_="card-header-title"),
               ui.div(ui.p("Mold Code 검색", style="font-weight:600;margin-bottom:12px;margin-top:16px"),
                      ui.div(id="mold-code-toggles", style="display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:20px")),
               ui.div(ui.p("변수 설정 (복수 선택 가능)", style="font-weight:600;margin-bottom:12px;margin-top:16px"),
                      ui.div(id="variable-toggles", style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px")),
               ui.tags.div(ui.input_radio_buttons("mold_code_select", None, choices=MOLD_CODES, selected="all", inline=False), style="display:none"),
               ui.tags.div(ui.input_checkbox_group("variable_select", None, choices=VARIABLES, selected=DEFAULT_VARIABLES, inline=False), style="display:none"),
               class_="custom-card"),
        ui.div(ui.div("런 차트", class_="card-header-title"),
               ui.output_ui("run_charts_container"),
               class_="custom-card"),
        ui.div(ui.div("Top 10 로그 (실시간 데이터)", class_="card-header-title"),
               ui.div(ui.output_data_frame("tab1_table_realtime"), class_="table-container"),
               class_="custom-card"),
        class_="main-container"),
    ui.tags.script("""
    $(document).ready(function(){
      var codes = ['all', '8412', '8917', '8722', '8413', '8576'];
      var toggleHTML = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px">';
      codes.forEach(function(code, idx){
        if(idx === 0){
          toggleHTML += '</div><div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px">';
        }
        toggleHTML += '<div style="text-align:center"><label style="display:flex;flex-direction:column;align-items:center;gap:6px;cursor:pointer;margin:0"><input type="radio" name="mold_toggle" value="' + code + '" ' + (idx === 0 ? 'checked' : '') + ' style="display:none"><div class="toggle-switch ' + (idx === 0 ? 'active' : '') + '"><div class="toggle-circle"></div></div></label><span style="font-size:11px">' + code + '</span></div>';
      });
      toggleHTML += '</div>';
      $('#mold-code-toggles').html(toggleHTML);
      
      $(document).on('change', 'input[name="mold_toggle"]', function(){
        var selectedValue = $(this).val();
        Shiny.setInputValue('mold_code_select', selectedValue);
        
        $('input[name="mold_toggle"]').each(function(){
          if(this.checked){
            $(this).closest('label').find('.toggle-switch').addClass('active');
          } else {
            $(this).closest('label').find('.toggle-switch').removeClass('active');
          }
        });
      });

      var variables = """ + str(list(VARIABLES.keys())) + """;
      var variableLabels = """ + str(list(VARIABLES.values())) + """;
      var defaultSelected = """ + str(DEFAULT_VARIABLES) + """;
      var variableToggleHTML = '';
      variables.forEach(function(varKey, idx){
        var labelText = variableLabels[idx].split(' (')[0];
        var isDefault = defaultSelected.includes(varKey);
        variableToggleHTML += '<div style="text-align:center"><label style="display:flex;flex-direction:column;align-items:center;gap:6px;cursor:pointer;margin:0"><input type="checkbox" name="variable_toggle" value="' + varKey + '" ' + (isDefault ? 'checked' : '') + ' style="display:none"><div class="toggle-switch ' + (isDefault ? 'active' : '') + '"><div class="toggle-circle"></div></div></label><span style="font-size:11px">' + labelText + '</span></div>';
      });
      $('#variable-toggles').html(variableToggleHTML);
      
      $(document).on('change', 'input[name="variable_toggle"]', function(){
        var selectedValues = [];
        $('input[name="variable_toggle"]:checked').each(function(){
          selectedValues.push($(this).val());
        });
        Shiny.setInputValue('variable_select', selectedValues);
        
        $('input[name="variable_toggle"]').each(function(){
          if(this.checked){
            $(this).closest('label').find('.toggle-switch').addClass('active');
          } else {
            $(this).closest('label').find('.toggle-switch').removeClass('active');
          }
        });
      });
    });
    """),
    ui.tags.script("""
    // 불량 센서 상태 업데이트 핸들러
    Shiny.addCustomMessageHandler('update_sensors', function(data) {
        var defectEl = document.getElementById('defect_indicator');
        
        if (defectEl) {
            defectEl.classList.remove('active-defect');
            if (data.is_defect) {
                defectEl.classList.add('active-defect');
            }
        }
    });
    """),
)

# SERVER
def tab_server(input, output, session, streamer, shared_df, streaming_active, defect_indicator=None, anomaly_indicator=None):
    
    @output
    @render.ui
    def kpi_yield():
        return ui.h1("95.5%", class_="kpi-value")
    
    # 불량 센서 상태 업데이트 (실시간)
    @reactive.effect
    def _sensor_update():
        reactive.invalidate_later(500)
        is_defect = defect_indicator() if defect_indicator else False
        
        # Shiny 메시지로 JavaScript에 전달
        session.send_custom_message("update_sensors", {
            "is_defect": is_defect
        })

    @output
    @render.ui
    def kpi_cycle():
        df = shared_df()
        if df.empty or 'production_cycletime' not in df.columns:
            return ui.h1("0.0 sec", class_="kpi-value")
        return ui.h1(f"{df['production_cycletime'].mean():.1f} sec", class_="kpi-value")

    @output
    @render.ui
    def kpi_uptime():
        df = shared_df()
        if df.empty or 'working' not in df.columns:
            return ui.h1("0.0%", class_="kpi-value")
        working = (df['working'] == '가동').sum()
        return ui.h1(f"{(working/len(df)*100):.1f}%", class_="kpi-value")

    def create_single_chart(df, mold_code, variable):
        """단일 차트를 생성하는 헬퍼 함수"""
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        def empty(title, msg='', ylabel='값', color='black'):
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=color)
            if msg:
                ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='#999', transform=ax.transAxes)
            plt.tight_layout()
            return fig
        
        label = get_label(variable)
        
        if df.empty or df is None:
            return empty(f'Mold Code {mold_code} - {label}', '데이터 로딩 중...', ylabel=label)
        
        try:
            if 'mold_code' not in df.columns:
                return empty(f'Mold Code {mold_code} - {label}', 'mold_code 컬럼 없음', label, '#e74c3c')
            
            mold_code_str = str(mold_code)
            df_mold = df['mold_code'].astype(str)
            
            if mold_code_str == "all":
                filtered = df.copy()
            else:
                filtered = df[df_mold == mold_code_str].copy()
            
            if filtered.empty:
                codes = sorted(df['mold_code'].astype(str).unique())
                return empty(f'{"모든 Mold Code" if mold_code_str == "all" else f"Mold Code {mold_code}"} - {label}', f'데이터 없음\n존재: {", ".join(codes)}', label, '#e74c3c')
            
            if variable not in filtered.columns:
                return empty(f'Mold Code {mold_code} - {label}', f'"{label}" 없음', label, '#e74c3c')
            
            if 'registration_time' in filtered.columns:
                filtered = filtered.sort_values('registration_time')
                plot_df = filtered[['registration_time', variable]].dropna().reset_index(drop=True).tail(10)
            else:
                plot_df = filtered[[variable]].dropna().reset_index(drop=True).tail(10)
            
            if plot_df.empty:
                return empty(f'Mold Code {mold_code} - {label}', '결측치만 존재', label, '#e74c3c')
            
            if 'registration_time' in plot_df.columns:
                x_time = pd.to_datetime(plot_df['registration_time'], errors='coerce')
                use_time_axis = True
            else:
                x_time = range(len(plot_df))
                use_time_axis = False
            
            y = plot_df[variable].values
            
            try:
                y = y.astype(float)
            except:
                return empty(f'Mold Code {mold_code} - {label}', '숫자 데이터 아님', label, '#e74c3c')
            
            ax.plot(x_time, y, color='#4A90E2', linewidth=2, marker='o', markersize=6,
                   markerfacecolor='#2c3e50', markeredgecolor='white', markeredgewidth=1.5)
            
            mean = y.mean()
            ax.axhline(y=mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'평균: {mean:.2f}')
            
            if use_time_axis:
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.tick_params(axis='x', labelsize=8)
            
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{"모든 Mold Code" if mold_code_str == "all" else f"Mold Code {mold_code}"} - {label}', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return empty('오류 발생', str(e)[:50], color='#e74c3c')

    @output
    @render.ui
    def run_charts_container():
        df = shared_df()
        mold_code = input.mold_code_select()
        variables = input.variable_select()
        
        if not variables:
            return ui.div(
                ui.p("변수를 선택하세요", style="text-align:center;padding:40px;color:#999;font-size:16px")
            )
        
        # 각 변수에 대해 차트 생성
        chart_outputs = []
        for var in variables:
            chart_id = f"chart_{var}"
            chart_outputs.append(
                ui.div(
                    ui.output_plot(chart_id, height="450px"),
                    class_="chart-container"
                )
            )
        
        return ui.div(*chart_outputs)
    
    # 동적으로 각 변수의 차트 렌더링
    @reactive.effect
    def _create_chart_renderers():
        variables = input.variable_select()
        if not variables:
            return
        
        df = shared_df()
        mold_code = input.mold_code_select()
        
        for var in variables:
            chart_id = f"chart_{var}"
            
            # 클로저 문제 해결을 위한 함수 팩토리
            def make_renderer(variable):
                @output(id=chart_id)
                @render.plot
                def _():
                    return create_single_chart(shared_df(), input.mold_code_select(), variable)
                return _
            
            make_renderer(var)

    @output
    @render.data_frame
    def tab1_table_realtime():
        df = shared_df()
        
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