from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import math
from pathlib import Path
import joblib
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

ALERT_VARIABLES = [
    ("cast_pressure", "주조 압력"),
    ("upper_mold_temp1", "상부 금형 온도1"),
    ("low_section_speed", "저속 구간 속도"),
    ("biscuit_thickness", "비스킷 두께"),
]

ALERT_STATUS_TEXT = {
    "normal": "정상",
    "warning": "WARNING (값 튐)",
    "drift": "DRIFT (추세 이상)",
    "critical": "CRITICAL (즉시 조치)",
    "no_data": "데이터 없음",
}

LEVEL_CIRCLE_CLASS = {
    "normal": "status-green",
    "warning": "status-warning",
    "drift": "status-drift",
    "critical": "status-critical",
    "no_data": "status-muted",
}

LEVEL_STATUS_CLASS = {
    "normal": "level-normal",
    "warning": "level-warning",
    "drift": "level-drift",
    "critical": "level-critical",
    "no_data": "level-no_data",
}

BASE_DIR = Path(__file__).resolve().parents[2]
BASELINE_FILE = BASE_DIR / "data" / "processed" / "train_v1_time.csv"

MAD_THRESHOLD = 2.5
EWMA_LAMBDA = 0.3
EWMA_REQUIRED_RUNS = 2
EWMA_SIGMA = math.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA)) if 0 < EWMA_LAMBDA < 1 else 1.0
EWMA_LIMIT = 2 * EWMA_SIGMA
EWMA_TRACKER = {}
ANOMALY_MAX_ROWS = 500


def load_baseline_stats(path: Path, variables):
    stats = {}
    if not path.exists():
        return stats
    try:
        df = pd.read_csv(path)
    except Exception:
        return stats

    keys = [key for key, _ in variables]
    for key in keys:
        if key not in df.columns:
            continue
        series = pd.to_numeric(df[key], errors="coerce").dropna()
        if series.empty:
            continue
        median = float(series.median())
        mad = float(np.median(np.abs(series - median)))
        if not np.isfinite(mad) or mad == 0.0:
            q75, q25 = series.quantile([0.75, 0.25])
            iqr = float(q75 - q25)
            if np.isfinite(iqr) and iqr != 0.0:
                mad = iqr / 1.349
        if not np.isfinite(mad) or mad == 0.0:
            std = float(series.std(ddof=1))
            if np.isfinite(std) and std != 0.0:
                mad = std / 1.4826
        if not np.isfinite(mad) or mad == 0.0:
            mad = 1.0
        stats[key] = {"median": median, "mad": max(mad, 1e-6)}
    return stats


BASELINE_STATS = load_baseline_stats(BASELINE_FILE, ALERT_VARIABLES)

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
.status-indicator-circle.status-green{background:#3B7D23;box-shadow:0 0 20px rgba(59,125,35,.45)}
.status-indicator-circle.status-warning{background:#f0ad4e;box-shadow:0 0 20px rgba(240,173,78,.5)}
.status-indicator-circle.status-drift{background:#3498db;box-shadow:0 0 20px rgba(52,152,219,.5)}
.status-indicator-circle.status-critical{background:#C00000;box-shadow:0 0 30px rgba(192,0,0,.6)}
.status-indicator-circle.status-muted{background:#bdc3c7;box-shadow:0 0 12px rgba(189,195,199,.45)}
.status-indicator-label{font-size:17px;font-weight:700;color:#111;white-space:nowrap;text-align:center}
.defect-overlay{position:fixed;top:24px;right:24px;width:260px;background:#fff;border-radius:16px;padding:18px;
box-shadow:0 10px 24px rgba(0,0,0,.18);display:none;flex-direction:column;gap:14px;z-index:9999;cursor:default}
.defect-overlay.visible{display:flex}
.overlay-header{display:flex;justify-content:space-between;align-items:center;font-weight:700;color:#111;cursor:move;user-select:none}
.overlay-close-btn{border:none;background:transparent;font-size:20px;line-height:1;cursor:pointer;color:#666}
.overlay-close-btn:hover{color:#000}
.overlay-light-list{display:flex;flex-direction:column;gap:12px}
.overlay-light-row{display:flex;align-items:center;gap:12px}
.overlay-light-row .status-indicator-circle{width:38px;height:38px;border:3px solid #1f1f1f;box-shadow:0 0 12px rgba(0,0,0,.15)}
.overlay-light-name{font-size:13px;font-weight:600;color:#2c3e50}
.overlay-light-text{display:flex;flex-direction:column;gap:2px}
.overlay-light-status{font-size:12px;font-weight:700;color:#7f8c8d}
.overlay-light-status.level-normal{color:#2ecc71}
.overlay-light-status.level-warning{color:#f39c12}
.overlay-light-status.level-drift{color:#2980b9}
.overlay-light-status.level-critical{color:#e74c3c}
.overlay-light-status.level-no_data{color:#7f8c8d}
.overlay-light-grid{display:flex;flex-wrap:wrap;gap:12px}
.overlay-light-grid .overlay-light-row{flex:1 1 48%;min-width:120px}
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
.fade-in{animation:fadeInEase .35s ease both}
@keyframes fadeInEase{0%{opacity:0;transform:translateY(6px)}100%{opacity:1;transform:translateY(0)}}
"""

# 헬퍼 함수
def create_kpi(title, output_id, subtitle, line_class):
    return ui.div(ui.p(title, class_="kpi-title"), ui.div(class_=f"kpi-line {line_class}"),
                  ui.output_ui(output_id), ui.p(subtitle, class_="kpi-sub"), class_="kpi-card")

def create_light(output_id, label):
    return ui.div(
        ui.output_ui(output_id),
        ui.div(label, class_="status-indicator-label"),
        class_="status-row"
    )

def get_label(variable):
    return VARIABLES.get(variable, variable).split(' (')[0]


def run_charts_container_ui():
    """Build a static container that toggles chart visibility without re-rendering the DOM."""
    no_selection_panel = ui.panel_conditional(
        "!(input.variable_select && input.variable_select.length)",
        ui.div(
            ui.p(
                "변수를 선택하세요",
                style="text-align:center;padding:40px;color:#999;font-size:16px",
            )
        ),
    )

    chart_panels = [
        ui.panel_conditional(
            f"input.variable_select && input.variable_select.includes('{variable}')",
            ui.div(
                ui.output_plot(f"chart_{variable}", height="450px"),
                class_="chart-container",
            ),
        )
        for variable in VARIABLES.keys()
    ]

    return ui.div(no_selection_panel, *chart_panels)


def overlay_light_row_ui(key: str, label: str):
    """Create a single overlay row with placeholders for circle and status outputs."""
    return ui.div(
        ui.output_ui(f"overlay_light_circle_{key}"),
        ui.div(
            ui.span(label, class_="overlay-light-name"),
            ui.output_ui(f"overlay_light_status_{key}"),
            class_="overlay-light-text",
        ),
        class_="overlay-light-row",
    )


def overlay_light_grid_static_ui():
    """Build the static overlay layout so only the inner outputs update over time."""
    rows = [overlay_light_row_ui("defect", "불량 발생")]
    rows.extend(
        overlay_light_row_ui(var_name, var_label)
        for var_name, var_label in ALERT_VARIABLES
    )
    return ui.div(*rows, class_="overlay-light-grid")


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
            ui.div(
                ui.div(
                    create_light("defect_indicator_ui", "불량 발생"),
                    ui.input_action_button("toggle_defect_overlay", "⚠ 경보 창 띄우기", class_="btn btn-outline-danger btn-sm"),
                    class_="status-panel"
                ),
                style="flex:1"
            ),
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
               run_charts_container_ui(),
               class_="custom-card"),
        ui.div(ui.div("Top 10 로그 (실시간 데이터)", class_="card-header-title"),
               ui.div(ui.output_data_frame("tab1_table_realtime"), class_="table-container"),
               class_="custom-card"),
        class_="main-container"),
    ui.div(
        ui.div("불량 경보", ui.tags.button("×", id="close_defect_overlay", class_="overlay-close-btn", **{"aria-label": "닫기"}), class_="overlay-header"),
        overlay_light_grid_static_ui(),
        id="defect-overlay",
        class_="defect-overlay"
    ),
    ui.tags.script("""
    $(document).ready(function(){
      var overlay = $('#defect-overlay');
      var isDragging = false;
      var dragOffset = {x:0,y:0};
      $(document).on('click', '#toggle_defect_overlay', function(){
        overlay.toggleClass('visible');
        if (overlay.hasClass('visible')){
          var storedLeft = overlay.data('pos-left');
          var storedTop = overlay.data('pos-top');
          if (storedLeft){
            overlay.css({left: storedLeft, top: storedTop || '24px', right: 'auto'});
          } else {
            overlay.css({top:'24px', right:'24px', left:'auto'});
          }
        } else {
          $(document).off('.defectDrag');
          isDragging = false;
        }
      });
      $(document).on('click', '#close_defect_overlay', function(){
        overlay.removeClass('visible');
        $(document).off('.defectDrag');
        isDragging = false;
      });
      $(document).on('mousedown', '#defect-overlay .overlay-header', function(e){
        if ($(e.target).closest('.overlay-close-btn').length){
          return;
        }
        if (!overlay.hasClass('visible')){
          return;
        }
        isDragging = true;
        var rect = overlay[0].getBoundingClientRect();
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;
        overlay.css({right:'auto'});
        $(document).on('mousemove.defectDrag', function(ev){
          if (!isDragging) return;
          var newLeft = ev.clientX - dragOffset.x;
          var newTop = ev.clientY - dragOffset.y;
          overlay.css({left: newLeft + 'px', top: newTop + 'px'});
        });
        $(document).on('mouseup.defectDrag', function(){
          if (!isDragging) return;
          isDragging = false;
          overlay.data('pos-left', overlay.css('left'));
          overlay.data('pos-top', overlay.css('top'));
          $(document).off('.defectDrag');
        });
        e.preventDefault();
      });
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
)

## SERVER
def tab_server(input, output, session, streamer, shared_df, streaming_active, defect_indicator=None, anomaly_indicator=None):
    
    @output
    @render.ui
    def kpi_yield():
        return ui.h1("95.5%", class_="kpi-value")
    
    # 불량 센서 상태 업데이트 (실시간)
    @reactive.calc
    def anomaly_summary():
        summary = {}
        df = shared_df.get()
        if df is None:
            df = pd.DataFrame()

        selected_code = input.mold_code_select()
        if selected_code is None or selected_code == "":
            selected_code = "all"
        selected_code = str(selected_code)

        if selected_code != "all" and "mold_code" in df.columns:
            df = df[df["mold_code"].astype(str) == selected_code]
        if df is None or df.empty:
            return summary

        if 0 < ANOMALY_MAX_ROWS < len(df):
            df = df.tail(ANOMALY_MAX_ROWS).copy()

        available_keys = [var_name for var_name, _ in ALERT_VARIABLES if var_name in df.columns]
        numeric_df = pd.DataFrame()
        if available_keys:
            numeric_df = df[available_keys].apply(pd.to_numeric, errors="coerce")

        for var_name, var_label in ALERT_VARIABLES:
            stats = BASELINE_STATS.get(var_name)
            default_entry = {
                "level": "no_data",
                "display": ALERT_STATUS_TEXT.get("no_data", "데이터 없음"),
                "latest": None,
                "median": None,
                "mad": None,
                "z": None,
                "ewma": None,
                "mad_breach": False,
                "drift": False,
                "mean": None,
                "std": None,
            }
            summary[var_name] = default_entry
            if stats is None or var_name not in numeric_df.columns:
                continue

            series = numeric_df[var_name].dropna()
            if series.empty:
                continue

            latest = float(series.iloc[-1])
            median = stats["median"]
            mad = stats["mad"]
            z_score = 0.6745 * (latest - median) / mad if mad else 0.0
            mad_breach = abs(z_score) > MAD_THRESHOLD

            tracker_key = (selected_code, var_name)
            tracker = EWMA_TRACKER.get(tracker_key, {"ewma": 0.0, "consec": 0})
            ewma_val = EWMA_LAMBDA * z_score + (1 - EWMA_LAMBDA) * tracker.get("ewma", 0.0)
            if not np.isfinite(ewma_val):
                ewma_val = 0.0
            ewma_breach = abs(ewma_val) > EWMA_LIMIT
            consec = tracker.get("consec", 0) + 1 if ewma_breach else 0
            drift = consec >= EWMA_REQUIRED_RUNS
            EWMA_TRACKER[tracker_key] = {"ewma": ewma_val, "consec": consec}

            if mad_breach:
                level = "critical"
                display_text = "CRITICAL (값 튐)" if not drift else "CRITICAL (즉시 조치)"
            elif drift:
                level = "warning"
                display_text = "WARNING (추세 이상)"
            else:
                level = "normal"
                display_text = "정상"

            mean_val = float(series.mean())
            if not np.isfinite(mean_val):
                mean_val = None
            std_val = float(series.std(ddof=1)) if len(series) > 1 else None
            if std_val is not None and not np.isfinite(std_val):
                std_val = None

            summary[var_name] = {
                "level": level,
                "display": display_text,
                "latest": latest,
                "median": median,
                "mad": mad,
                "z": z_score,
                "ewma": ewma_val,
                "mad_breach": mad_breach,
                "drift": drift,
                "mean": mean_val,
                "std": std_val,
            }
        return summary

    @reactive.calc
    def defect_state():
        state = {
            "level": "no_data",
            "text": ALERT_STATUS_TEXT.get("no_data", "데이터 없음"),
            "tooltip": None,
        }
        df = shared_df.get()
        if df is None or df.empty:
            return state

        is_defect = False
        level = "normal"
        text = "정상"
        tooltip = "정상 동작 중"

        if defect_indicator:
            raw = defect_indicator()
            if isinstance(raw, dict):
                is_defect = bool(raw.get("is_defect", False))
                level = raw.get("level") or ("critical" if is_defect else "normal")
                text = raw.get("text") or ("CRITICAL (불량 예측)" if is_defect else "정상")
                tooltip = raw.get("tooltip") or ("불량으로 예측됨" if is_defect else "정상 동작 중")
            else:
                is_defect = bool(raw)
                level = "critical" if is_defect else "normal"
                text = "CRITICAL (불량 예측)" if is_defect else "정상"
                tooltip = "불량으로 예측됨" if is_defect else "정상 동작 중"
        else:
            # defect_indicator가 없으면 기본 정상
            is_defect = False
            level = "normal"
            text = "정상"
            tooltip = "정상 동작 중"

        if level not in LEVEL_CIRCLE_CLASS:
            level = "critical" if is_defect else "normal"

        if not text:
            text = "CRITICAL (불량 예측)" if level == "critical" else ALERT_STATUS_TEXT.get(level, "정상")
        if not tooltip:
            tooltip = "불량으로 예측됨" if level == "critical" else "정상 동작 중"

        state.update({
            "level": level,
            "text": text,
            "tooltip": tooltip,
        })
        return state

    @output
    @render.ui
    def defect_indicator_ui():
        state = defect_state()
        level = state.get("level", "no_data")
        circle_class = LEVEL_CIRCLE_CLASS.get(level, "status-muted")
        tooltip = state.get("tooltip")
        div_kwargs = {
            "id": "defect_indicator",
            "class_": f"status-indicator-circle {circle_class}"
        }
        if tooltip:
            div_kwargs["title"] = tooltip
        return ui.div(**div_kwargs)

    @output
    @render.ui
    def overlay_light_circle_defect():
        state = defect_state()
        level = state.get("level", "no_data")
        circle_class = LEVEL_CIRCLE_CLASS.get(level, "status-muted")
        tooltip = state.get("tooltip")
        attrs = {"class_": f"status-indicator-circle {circle_class}"}
        if tooltip:
            attrs["title"] = tooltip
        return ui.div(**attrs)

    @output
    @render.ui
    def overlay_light_status_defect():
        state = defect_state()
        level = state.get("level", "no_data")
        text = state.get("text") or ALERT_STATUS_TEXT.get(level, "데이터 없음")
        status_class = LEVEL_STATUS_CLASS.get(level, "level-no_data")
        return ui.span(text, class_=f"overlay-light-status {status_class}")

    def build_anomaly_tooltip(info: dict | None):
        if not info:
            return None
        tooltip_parts = []
        latest = info.get("latest")
        if isinstance(latest, (int, float)) and np.isfinite(latest):
            tooltip_parts.append(f"최근값: {latest:.2f}")
        flags = []
        if info.get("mad_breach"):
            flags.append("MAD 경보")
        if info.get("drift"):
            flags.append("EWMA 경보")
        if flags:
            tooltip_parts.append(", ".join(flags))
        return "\n".join(part for part in tooltip_parts if part) or None

    def register_overlay_alert_outputs(var_name: str, var_label: str):
        circle_id = f"overlay_light_circle_{var_name}"
        status_id = f"overlay_light_status_{var_name}"

        @output(id=circle_id)
        @render.ui
        def _circle(var_key=var_name):
            info = anomaly_summary().get(var_key, {})
            level = info.get("level", "no_data")
            circle_class = LEVEL_CIRCLE_CLASS.get(level, "status-muted")
            tooltip = build_anomaly_tooltip(info)
            attrs = {"class_": f"status-indicator-circle {circle_class}"}
            if tooltip:
                attrs["title"] = tooltip
            return ui.div(**attrs)

        @output(id=status_id)
        @render.ui
        def _status(var_key=var_name):
            info = anomaly_summary().get(var_key, {})
            level = info.get("level", "no_data")
            text = info.get("display") or ALERT_STATUS_TEXT.get(level, "데이터 없음")
            status_class = LEVEL_STATUS_CLASS.get(level, "level-no_data")
            return ui.span(text, class_=f"overlay-light-status {status_class}")

    for var_name, var_label in ALERT_VARIABLES:
        register_overlay_alert_outputs(var_name, var_label)

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

    def create_single_chart(df, mold_code, variable, statuses=None, defect_info=None):
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

            highlight_idx = []
            status_info = (statuses or {}).get(variable) if statuses else None
            if status_info and status_info.get("level") in {"warning", "drift", "critical"} and len(y) > 0:
                highlight_idx.append(len(y) - 1)
            if defect_info and defect_info.get("level") in {"critical"} and len(y) > 0:
                highlight_idx.append(len(y) - 1)
            if highlight_idx:
                highlight_idx = sorted(set(highlight_idx))
                x_highlight = [list(x_time)[i] for i in highlight_idx]
                y_highlight = y[highlight_idx]
                ax.scatter(
                    x_highlight,
                    y_highlight,
                    color='#C00000',
                    edgecolor='white',
                    s=100,
                    zorder=5,
                    label='알림 발생 지점'
                )
            
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

    def register_chart_output(variable_key: str):
        chart_id = f"chart_{variable_key}"

        @output(id=chart_id)
        @render.plot
        def _():
            selected_variables = input.variable_select() or []
            if variable_key not in selected_variables:
                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.axis("off")
                plt.tight_layout()
                return fig

            df_local = shared_df.get()
            statuses = anomaly_summary()
            defect_info = defect_state()
            return create_single_chart(
                df_local,
                input.mold_code_select(),
                variable_key,
                statuses=statuses,
                defect_info=defect_info,
            )

        return _

    for variable_key in VARIABLES.keys():
        register_chart_output(variable_key)

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
