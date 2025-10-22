# ===========================
# modules/tab_target_operation_manager.py
# (OTF 전용 한글폰트 로더 적용 버전: 배포시 한글 깨짐 방지)
# ===========================

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

# ---------- [추가] 내장 모델 로더/예측기 (app.py 수정 불필요) ----------
import os, json, pickle, threading

# (선택) Matplotlib 캐시를 앱 내부에 두고 싶다면 주석 해제
# MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / "assets" / ".mplcache"
# os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
# try:
#     MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# except Exception as e:
#     print(f"[FONT] MPLCONFIGDIR 만들기 실패: {e}")

class _TabModel:
    """게으른 로딩 + 싱글톤(배포 친화). 환경변수로 경로/피처 오버라이드."""
    _inst = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._inst is None:
            with cls._lock:
                if cls._inst is None:
                    cls._inst = super().__new__(cls)
                    cls._inst._inited = False
        return cls._inst

    def __init__(self):
        if getattr(self, "_inited", False):
            return
        self._model = None
        self._feature_cols = None
        self._inited = True
        self._load_once()

    def _load_once(self):
        here = Path(__file__).resolve().parent
        project_root = here.parents[1]

        default_model = project_root / "data" / "models" / "LightGBM_v1.pkl"
        model_path = Path(os.getenv("MODEL_PATH", default_model))

        default_feats = [
            "molten_temp","facility_operation_cycleTime","production_cycletime",
            "low_section_speed","high_section_speed","molten_volume","cast_pressure",
            "biscuit_thickness","upper_mold_temp1","upper_mold_temp2","upper_mold_temp3",
            "lower_mold_temp1","lower_mold_temp2","lower_mold_temp3",
            "sleeve_temperature","physical_strength","Coolant_temperature","EMS_operation_time"
        ]
        feats_env = os.getenv("FEATURE_COLS_JSON", "")
        if feats_env:
            try:
                self._feature_cols = json.loads(feats_env)
            except Exception as e:
                print(f"[MODEL] FEATURE_COLS_JSON 파싱 실패 → 기본 사용: {e}")
                self._feature_cols = default_feats
        else:
            self._feature_cols = default_feats

        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            print(f"[MODEL] loaded in tab module: {model_path}")
        except Exception as e:
            print(f"[MODEL] load failed: {model_path} -> {e}")
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None and bool(self._feature_cols)

    def predict_row(self, row: pd.Series):
        """row → {'prob': float} 또는 {'is_defect': bool}. 실패 시 None."""
        if not self.ready:
            return None
        try:
            x = pd.to_numeric(row.reindex(self._feature_cols), errors="coerce") \
                    .astype(float).fillna(0.0).values.reshape(1, -1)
            m = self._model
            if hasattr(m, "predict_proba"):
                prob = float(m.predict_proba(x)[0, 1])
                return {"prob": prob}
            elif hasattr(m, "decision_function"):
                score = float(m.decision_function(x)[0])
                return {"is_defect": score > 0.0}
            else:
                yhat = int(m.predict(x)[0])
                return {"is_defect": bool(yhat)}
        except Exception as e:
            print(f"[MODEL] predict error: {e}")
            return None

_MODEL_SINGLETON = _TabModel()

def _predict_row_from_model(row: pd.Series):
    return _MODEL_SINGLETON.predict_row(row)
# ---------------------------------------------------------------------

# ===========================
# 한글 폰트: OTF만 로드(요청안)
# ===========================
def setup_korean_font_otf_only():
    """
    번들된 OTF만 사용. 성공 시 해당 family를 전역 기본 폰트로 지정.
    실패하면 경고만 찍고(DejaVu 유지) 계속 진행.
    """
    try:
        base_dir = Path(__file__).resolve().parents[1]  # dashboard/
        otf_candidates = [
            base_dir / "assets" / "fonts" / "NanumGothic.otf",
            base_dir / "assets" / "fonts" / "NotoSansKR-Regular.otf",
        ]
        for fp in otf_candidates:
            if fp.exists():
                fm.fontManager.addfont(str(fp))
                # 캐시 갱신
                try:
                    fm._load_fontmanager(try_read_cache=False)
                except Exception:
                    pass
                family = fm.FontProperties(fname=str(fp)).get_name()
                plt.rcParams["font.family"] = family
                plt.rcParams["font.sans-serif"] = [family]
                plt.rcParams["axes.unicode_minus"] = False
                # 실제 사용 가능 확인
                fm.findfont(family, fallback_to_default=False)
                print(f"[FONT] Using bundled OTF: {fp} -> family='{family}'")
                return
        print("[FONT] No bundled OTF found or not usable. Korean may break.")
    except Exception as e:
        print(f"[FONT] OTF load failed: {e}")

# 초기 한 번 적용
setup_korean_font_otf_only()

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

# 기본 선택 변수
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

# 프로젝트 루트(dashboard/) 기준 경로
BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_FILE = BASE_DIR / "data" / "processed" / "train_v1_time.csv"
# ★ test 예측파일(라벨 포함) 경로 - 선적재는 하지 않고, 실시간 매칭에만 사용
TEST_PRED_FILE = BASE_DIR / "data" / "intermin" / "test_predictions_v2.csv"

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
.var-group{background:#fafbfc;border:1px solid #e6e8eb;border-radius:12px;padding:14px 16px}
.var-group + .var-group{margin-top:12px}
.var-group-title{font-weight:700;color:#2A2D30;margin-bottom:10px;font-size:14px}
.var-group-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:16px}
@media (max-width:1200px){.var-group-grid{grid-template-columns:repeat(4,1fr)}}
@media (max-width:992px){.var-group-grid{grid-template-columns:repeat(3,1fr)}}
@media (max-width:768px){.var-group-grid{grid-template-columns:repeat(2,1fr)}}
"""

# 헬퍼
def create_kpi(title, output_id, subtitle, line_class):
    return ui.div(
        ui.p(title, class_="kpi-title"),
        ui.div(class_=f"kpi-line {line_class}"),
        ui.output_ui(output_id),
        ui.p(subtitle, class_="kpi-sub"),
        class_="kpi-card"
    )

def create_light(output_id, label):
    return ui.div(
        ui.output_ui(output_id),
        ui.div(label, class_="status-indicator-label"),
        class_="status-row"
    )

def get_label(variable):
    return VARIABLES.get(variable, variable).split(' (')[0]

def run_charts_container_ui():
    no_selection_panel = ui.panel_conditional(
        "!(input.variable_select && input.variable_select.length)",
        ui.div(
            ui.p("변수를 선택하세요",
                 style="text-align:center;padding:40px;color:#999;font-size:16px")
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
            ui.div(
                ui.layout_columns(
                    create_kpi("금형별 수율", "kpi_yield", "실시간 집계", "red-line"),
                    create_kpi("제품 사이클 타임", "kpi_cycle", "평균 사이클 타임", "yellow-line"),
                    create_kpi("설비 가동률", "kpi_uptime", "현재 가동 상태", "navy-line"),
                    col_widths=[4, 4, 4]
                ),
                style="flex:3"
            ),
            ui.div(
                ui.div(
                    create_light("defect_indicator_ui", "불량 발생"),
                    ui.input_action_button("toggle_defect_overlay", "⚠ 경보 창 띄우기",
                                           class_="btn btn-outline-danger btn-sm"),
                    class_="status-panel"
                ),
                style="flex:1"
            ),
            style="display:flex;gap:24px;margin-bottom:24px"
        ),
        ui.div(
            ui.div("검색 및 설정", class_="card-header-title"),
            ui.div(
                ui.p("Mold Code 검색", style="font-weight:600;margin-bottom:12px;margin-top:16px"),
                ui.div(id="mold-code-toggles",
                       style="display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:20px")
            ),
            ui.div(
                ui.p("변수 설정 (복수 선택 가능)", style="font-weight:600;margin-bottom:12px;margin-top:16px"),
                ui.div(id="variable-toggles", style="display:block")
            ),
            ui.tags.div(
                ui.input_radio_buttons("mold_code_select", None, choices=MOLD_CODES,
                                       selected="all", inline=False),
                style="display:none"
            ),
            ui.tags.div(
                ui.input_checkbox_group("variable_select", None, choices=VARIABLES,
                                        selected=DEFAULT_VARIABLES, inline=False),
                style="display:none"
            ),
            class_="custom-card"
        ),
        ui.div(
            ui.div("런 차트", class_="card-header-title"),
            run_charts_container_ui(),
            class_="custom-card"
        ),
        ui.div(
            ui.div("Top 10 로그 (실시간 데이터)", class_="card-header-title"),
            ui.div(ui.output_data_frame("tab1_table_realtime"), class_="table-container"),
            class_="custom-card"
        ),
        class_="main-container"
    ),
    ui.div(
        ui.div(
            "불량 경보",
            ui.tags.button("×", id="close_defect_overlay", class_="overlay-close-btn",
                           **{"aria-label": "닫기"}),
            class_="overlay-header"
        ),
        overlay_light_grid_static_ui(),
        id="defect-overlay",
        class_="defect-overlay"
    ),
    ui.tags.script("""
    $(document).ready(function(){

      // ====== (A) 경보 오버레이 드래그/토글 ======
      var overlay = $('#defect-overlay');
      var isDragging = false;
      var dragOffset = {x:0,y:0};

      $(document).on('click', '#toggle_defect_overlay', function(){
        overlay.toggleClass('visible');
        if (overlay.hasClass('visible')){
          var storedLeft = overlay.data('pos-left');
          var storedTop  = overlay.data('pos-top');
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
        if ($(e.target).closest('.overlay-close-btn').length){ return; }
        if (!overlay.hasClass('visible')){ return; }
        isDragging = true;
        var rect = overlay[0].getBoundingClientRect();
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;
        overlay.css({right:'auto'});
        $(document).on('mousemove.defectDrag', function(ev){
          if (!isDragging) return;
          var newLeft = ev.clientX - dragOffset.x;
          var newTop  = ev.clientY - dragOffset.y;
          overlay.css({left: newLeft + 'px', top: newTop + 'px'});
        });
        $(document).on('mouseup.defectDrag', function(){
          if (!isDragging) return;
          isDragging = false;
          overlay.data('pos-left', overlay.css('left'));
          overlay.data('pos-top',  overlay.css('top'));
          $(document).off('.defectDrag');
        });
        e.preventDefault();
      });

      // ====== (B) Mold code 토글 ======
      var codes = ['all', '8412', '8917', '8722', '8413', '8576'];
      var toggleHTML = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px">';
      codes.forEach(function(code, idx){
        if(idx === 0){
          toggleHTML += '</div><div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px">';
        }
        toggleHTML += ''
          + '<div style="text-align:center">'
          +   '<label style="display:flex;flex-direction:column;align-items:center;gap:6px;cursor:pointer;margin:0">'
          +     '<input type="radio" name="mold_toggle" value="' + code + '" ' + (idx === 0 ? 'checked' : '') + ' style="display:none">'
          +     '<div class="toggle-switch ' + (idx === 0 ? 'active' : '') + '"><div class="toggle-circle"></div></div>'
          +   '</label>'
          +   '<span style="font-size:11px">' + code + '</span>'
          + '</div>';
      });
      toggleHTML += '</div>';
      $('#mold-code-toggles').html(toggleHTML);

      $(document).on('change', 'input[name="mold_toggle"]', function(){
        var selectedValue = $(this).val();
        Shiny.setInputValue('mold_code_select', selectedValue);
        $('input[name="mold_toggle"]').each(function(){
          $(this).closest('label').find('.toggle-switch').toggleClass('active', this.checked);
        });
      });

      // ====== (C) 변수 토글: 녹이기/붓기/냉각 ======

      if (!document.getElementById('var-groups-css')){
        var style = document.createElement('style');
        style.id = 'var-groups-css';
        style.textContent = `
          .var-groups-wrapper{
            display:grid;
            grid-template-columns:repeat(3, minmax(0, 1fr));
            gap:16px;
            width:100%;
            box-sizing:border-box;
            align-items:stretch;
            grid-auto-rows:1fr;
          }
          @media (max-width:1280px){ .var-groups-wrapper{ grid-template-columns:repeat(2, minmax(0, 1fr)) } }
          @media (max-width:820px){  .var-groups-wrapper{ grid-template-columns:1fr } }

          .var-group{
            background:#fff;border:1px solid #e6e8eb;border-radius:16px;
            padding:18px 16px 14px;box-shadow:0 2px 8px rgba(0,0,0,.06);
            width:100%;max-width:100%;overflow:hidden;box-sizing:border-box;
            display:flex;flex-direction:column;height:100%;
          }
          .var-group + .var-group{ margin-top:0 !important; }

          .var-group-title{
            font-weight:800;color:#2A2D30;margin-bottom:12px;font-size:16px;flex:0 0 auto;
          }

          .var-group-grid{
            display:grid;
            grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
            gap:12px 14px;
            width:100%;
            box-sizing:border-box;
            flex:1 1 auto;
          }

          .var-item{
            display:flex;flex-direction:column;align-items:center;justify-content:flex-start;
            gap:8px;text-align:center;padding:10px 8px;min-height:105px;border-radius:12px;
            background:#fafbfc;border:1px solid #f0f2f4;overflow:hidden;
          }

          .toggle-switch{width:46px;height:28px;border-radius:14px}
          .toggle-switch .toggle-circle{width:20px;height:20px;top:4px;left:4px}
          .toggle-switch.active .toggle-circle{left:22px}
        `;
        document.head.appendChild(style);
      }

      var variables       = """ + str(list(VARIABLES.keys())) + """;
      var variableLabels  = """ + str(list(VARIABLES.values())) + """;
      var defaultSelected = """ + str(DEFAULT_VARIABLES) + """;

      var labelMap = {};
      variables.forEach(function(k, idx){ labelMap[k] = (variableLabels[idx]||'').split(' (')[0]; });

      var GROUPS = [
        { title: '녹이기',
          keys: ['molten_temp','molten_volume','sleeve_temperature'] },
        { title: '붓기',
          keys: ['cast_pressure','low_section_speed','high_section_speed','biscuit_thickness','EMS_operation_time','facility_operation_cycleTime','production_cycletime'] },
        { title: '냉각',
          keys: ['upper_mold_temp1','upper_mold_temp2','upper_mold_temp3','lower_mold_temp1','lower_mold_temp2','lower_mold_temp3','Coolant_temperature','physical_strength'] }
      ];

      var html = '<div class="var-groups-wrapper">';
      GROUPS.forEach(function(g){
        html += '<div class="var-group">';
        html +=   '<div class="var-group-title">' + g.title + '</div>';
        html +=   '<div class="var-group-grid">';
        g.keys.forEach(function(varKey){
          if(!(varKey in labelMap)) return;
          var isDefault = defaultSelected.includes(varKey);
          html += ''
            + '<div class="var-item">'
            +   '<label style="display:flex;flex-direction:column;align-items:center;gap:6px;cursor:pointer;margin:0">'
            +     '<input type="checkbox" name="variable_toggle" value="' + varKey + '" ' + (isDefault ? 'checked' : '') + ' style="display:none">'
            +     '<div class="toggle-switch ' + (isDefault ? 'active' : '') + '"><div class="toggle-circle"></div></div>'
            +   '</label>'
            +   '<span class="var-label">' + labelMap[varKey] + '</span>'
            + '</div>';
        });
        html +=   '</div>';
        html += '</div>';
      });
      html += '</div>';

      $('#variable-toggles').html(html);

      Shiny.setInputValue('variable_select', defaultSelected);

      $(document).on('change', 'input[name="variable_toggle"]', function(){
        var selectedValues = [];
        $('input[name="variable_toggle"]:checked').each(function(){
          selectedValues.push($(this).val());
        });
        Shiny.setInputValue('variable_select', selectedValues);
        $('input[name="variable_toggle"]').each(function(){
          $(this).closest('label').find('.toggle-switch').toggleClass('active', this.checked);
        });
      });

    });
    """),
)

from collections import defaultdict
import hashlib

# =====================
# ★ 금형별 수율: 선적재 + 실시간 누적
# =====================

# ✅ 금형별 수율 카운터
YIELD_CNT = defaultdict(lambda: {"good": 0, "total": 0})

def _row_fingerprint(s: pd.Series) -> str | None:
    """지문: registration_time, mold_code, count, id 를 우선 결합. 없으면 row 전체를 json으로."""
    if s is None or s.empty:
        return None
    keys = [k for k in ["registration_time", "mold_code", "count", "id"] if k in s.index]
    payload = "|".join(str(s[k]) for k in keys) if keys else s.to_json()
    try:
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    except Exception:
        return None

def _add_outcome(mold_code: str, outcome: int | None):
    if outcome is None:
        return
    YIELD_CNT[mold_code]["total"] += 1
    if outcome == 0:  # 정상
        YIELD_CNT[mold_code]["good"] += 1
    # 전체(all) 동시 집계
    YIELD_CNT["all"]["total"] += 1
    if outcome == 0:
        YIELD_CNT["all"]["good"] += 1

def initialize_yield_from_train():
    """train_v1_time.csv 의 passorfail 로 선적재(8413/8576은 train에 없으니 0에서 시작)."""
    try:
        df = pd.read_csv(BASELINE_FILE)
        if "mold_code" not in df.columns or "passorfail" not in df.columns:
            print("[초기화 경고] train에 mold_code/passorfail 없음")
            return
        df = df[["mold_code", "passorfail"]].copy()
        df["passorfail"] = pd.to_numeric(df["passorfail"], errors="coerce")
        df = df.dropna(subset=["passorfail"])
        df["passorfail"] = df["passorfail"].astype(int)

        for code, sub in df.groupby("mold_code"):
            code = str(code)
            total = len(sub)
            good  = int((sub["passorfail"] == 0).sum())
            YIELD_CNT[code]["total"] += total
            YIELD_CNT[code]["good"]  += good
            YIELD_CNT["all"]["total"] += total
            YIELD_CNT["all"]["good"]  += good

        print(f"[초기 수율 적재 완료] 금형 수: {len([k for k in YIELD_CNT if k!='all'])}, 전체: {YIELD_CNT['all']}")
    except Exception as e:
        print(f"[초기 수율 적재 실패] {e}")

initialize_yield_from_train()

# ✅ test_predictions_v2.csv 의 passorfail 매핑(선적재 X, 실시간 매칭에만 사용)
TEST_LABEL_MAP: dict[str, int] = {}
def _load_test_label_map():
    global TEST_LABEL_MAP
    TEST_LABEL_MAP = {}
    try:
        if not TEST_PRED_FILE.exists():
            print(f"[test 라벨] 파일 없음: {TEST_PRED_FILE}")
            return
        tdf = pd.read_csv(TEST_PRED_FILE)
        for _, r in tdf.iterrows():
            try:
                fp = _row_fingerprint(r)
                if not fp:
                    continue
                pf = pd.to_numeric(r.get("passorfail", np.nan), errors="coerce")
                if not np.isfinite(pf):
                    continue
                TEST_LABEL_MAP[fp] = int(pf)  # 0/1
            except Exception:
                continue
        print(f"[test 라벨] 매핑 준비 완료: {len(TEST_LABEL_MAP)} rows")
    except Exception as e:
        print(f"[test 라벨] 로드 실패: {e}")

_load_test_label_map()

# ✅ 서버
def tab_server(
    input, output, session,
    streamer, shared_df, streaming_active,
    defect_indicator=None,              # 프레임 전역 경보(ALL용)
    anomaly_indicator=None,
    predict_row=None,                   # 행 단위 예측 함수
    pred_col=None,                      # 0/1 라벨 칼럼
    proba_col=None,                     # 확률 칼럼
    proba_thresh: float = 0.5,          # 임계값
    predict_latest_for_code=None,       # [신규] 코드별 최신 예측 함수
):
    # --- [추가] app.py가 훅을 안 넘기면, 모듈 내장 모델 사용 ---
    if predict_row is None:
        predict_row = _predict_row_from_model

    _last_rows = {"n": 0, "last_fp": None}
    yield_tick = reactive.Value(0)
    # 모델 경보 전용 트리거: 선택 코드 새 데이터일 때만 증가
    alert_tick = reactive.Value(0)

    @reactive.effect
    @reactive.event(shared_df)
    def _accumulate_yield_realtime():
        df = shared_df.get()
        if df is None or df.empty or "mold_code" not in df.columns:
            print("[YIELD RT] 데이터 없음 또는 mold_code 없음")
            return

        if "registration_time" in df.columns:
            df = df.sort_values("registration_time")

        n = len(df)
        cur_fp = _row_fingerprint(df.iloc[-1]) if n > 0 else None
        is_new = (n > _last_rows.get("n", 0)) or (cur_fp and cur_fp != _last_rows.get("last_fp"))

        start = _last_rows.get("n", 0)
        if n > start:
            new_chunk = df.iloc[start:n]
        else:
            new_chunk = df.tail(1) if is_new else pd.DataFrame()

        for _, row in new_chunk.iterrows():
            mcode = str(row.get("mold_code", "unknown"))
            outcome = None

            # --- (1) test_predictions 매칭이 최우선 ---
            try:
                fp = _row_fingerprint(row)
                if fp and fp in TEST_LABEL_MAP:
                    outcome = int(TEST_LABEL_MAP[fp])  # 0(정상)/1(불량)
            except Exception:
                pass

            # --- (2) fallback: 기존 라벨/예측/확률/모델 ---
            if outcome is None and "passorfail" in row.index and pd.notna(row["passorfail"]):
                try:
                    v = int(pd.to_numeric(row["passorfail"], errors="coerce"))
                    if v in (0, 1):
                        outcome = v
                except:
                    pass

            if outcome is None and pred_col and (pred_col in row.index):
                try:
                    pv = pd.to_numeric(row[pred_col], errors="coerce")
                    if np.isfinite(pv):
                        outcome = int(pv) if pv in (0, 1) else int(float(pv) >= float(proba_thresh))
                except:
                    pass

            if outcome is None and proba_col and (proba_col in row.index):
                try:
                    prob = float(pd.to_numeric(row[proba_col], errors="coerce"))
                    if np.isfinite(prob):
                        outcome = int(prob >= float(proba_thresh))
                except:
                    pass

            if outcome is None and callable(predict_row):
                try:
                    pr = predict_row(row)
                    if isinstance(pr, dict):
                        if "is_defect" in pr:
                            outcome = 1 if bool(pr["is_defect"]) else 0
                        elif "prob" in pr:
                            outcome = 1 if float(pr["prob"]) >= float(proba_thresh) else 0
                    elif isinstance(pr, (int, np.integer, float, np.floating)):
                        outcome = 1 if float(pr) >= float(proba_thresh) else 0
                    else:
                        outcome = 1 if bool(pr) else 0
                except:
                    pass

            if outcome is None and defect_indicator is not None:
                try:
                    raw = defect_indicator()
                    is_defect = bool(raw.get("is_defect", False)) if isinstance(raw, dict) else bool(raw)
                    outcome = 1 if is_defect else 0
                except:
                    pass

            # --- (3) 누적 집계 ---
            _add_outcome(mcode, outcome)

        _last_rows["n"] = n
        _last_rows["last_fp"] = cur_fp

        if not new_chunk.empty or is_new:
            # KPI는 항상 갱신
            yield_tick.set(yield_tick.get() + 1)

            # 선택 코드 새 데이터가 있으면 모델 경보만 갱신
            try:
                selected_code = str(input.mold_code_select() or "all")
            except Exception:
                selected_code = "all"
            should_update_alert = False
            if selected_code == "all":
                should_update_alert = True
            else:
                if not new_chunk.empty and "mold_code" in new_chunk.columns:
                    should_update_alert = any(
                        str(mc) == selected_code for mc in new_chunk["mold_code"].astype(str)
                    )
                elif is_new and "mold_code" in df.columns:
                    last_mc = df.iloc[-1].get("mold_code", None)
                    last_mc = str(last_mc) if last_mc is not None and pd.notna(last_mc) else None
                    should_update_alert = (last_mc == selected_code)

            if should_update_alert:
                alert_tick.set(alert_tick.get() + 1)

    # KPI
    @output
    @render.ui
    def kpi_yield():
        _ = yield_tick.get()
        code = str(input.mold_code_select() or "all")
        rate = 0.0
        label = "전체" if code == "all" else f"Mold {code}"
        stats = YIELD_CNT.get(code, {"good": 0, "total": 0})
        total = stats["total"]
        good = stats["good"]
        if total > 0:
            rate = (good / total) * 100.0
        return ui.h1(f"{label}: {rate:.1f}%", class_="kpi-value")

    @output
    @render.ui
    def kpi_cycle():
        df = shared_df.get()
        if df is None or df.empty or 'production_cycletime' not in df.columns:
            return ui.h1("0.0 sec", class_="kpi-value")
        try:
            return ui.h1(f"{pd.to_numeric(df['production_cycletime'], errors='coerce').dropna().mean():.1f} sec", class_="kpi-value")
        except Exception:
            return ui.h1("0.0 sec", class_="kpi-value")

    @output
    @render.ui
    def kpi_uptime():
        df = shared_df.get()
        if (
            df is None or df.empty or
            'facility_operation_cycleTime' not in df.columns or
            'production_cycletime' not in df.columns
        ):
            return ui.h1("0.0%", class_="kpi-value")

        selected_code = str(input.mold_code_select() or "all")
        if selected_code != "all" and 'mold_code' in df.columns:
            df = df[df['mold_code'].astype(str) == selected_code]

        if df is None or df.empty:
            return ui.h1("0.0%", class_="kpi-value")

        prod_time = pd.to_numeric(df["production_cycletime"], errors="coerce").replace(0, np.nan)
        up = pd.to_numeric(df["facility_operation_cycleTime"], errors="coerce")

        ratio = prod_time / up
        uptime = np.nanmean(ratio) * 100
        if not np.isfinite(uptime):
            uptime = 0.0

        label = "전체" if selected_code == "all" else f"Mold {selected_code}"
        return ui.h1(f"{label}: {uptime:.1f}%", class_="kpi-value")

    # === (A) 이상치 경보: 기존 로직 ===
    @reactive.calc
    def anomaly_summary():
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
            return {}

        if 0 < ANOMALY_MAX_ROWS < len(df):
            df = df.tail(ANOMALY_MAX_ROWS).copy()

        available_keys = [var_name for var_name, _ in ALERT_VARIABLES if var_name in df.columns]
        numeric_df = pd.DataFrame()
        if available_keys:
            numeric_df = df[available_keys].apply(pd.to_numeric, errors="coerce")

        summary = {}
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

    # === (B) 모델 불량 경보: 선택 코드 새 데이터(alert_tick)일 때만 재계산 ===
    @reactive.calc
    def defect_state():
        _ = alert_tick.get()  # 선택 코드 관련 새 데이터가 들어올 때만 갱신

        def _state(level, text, tooltip):
            return {"level": level, "text": text, "tooltip": tooltip}

        df = shared_df.get()
        if df is None or df.empty:
            return _state("no_data", ALERT_STATUS_TEXT.get("no_data", "데이터 없음"), None)

        selected_code = input.mold_code_select()
        if not selected_code:
            selected_code = "all"
        selected_code = str(selected_code)

        # ALL: 전역 프레임 신호 그대로
        if selected_code == "all":
            if defect_indicator is None:
                return _state("normal", "정상", "정상 동작 중")
            raw = defect_indicator()
            if isinstance(raw, dict):
                is_defect = bool(raw.get("is_defect", False))
                level = raw.get("level") or ("critical" if is_defect else "normal")
                text  = raw.get("text")  or ("CRITICAL (불량 예측)" if is_defect else "정상")
                tip   = raw.get("tooltip") or ("불량으로 예측됨" if is_defect else "정상 동작 중")
            else:
                is_defect = bool(raw)
                level = "critical" if is_defect else "normal"
                text  = "CRITICAL (불량 예측)" if is_defect else "정상"
                tip   = "불량으로 예측됨" if is_defect else "정상 동작 중"
            if level not in LEVEL_CIRCLE_CLASS:
                level = "critical" if is_defect else "normal"
            return _state(level, text, tip)

        # 특정 코드
        if "mold_code" not in df.columns:
            return _state("no_data", ALERT_STATUS_TEXT.get("no_data", "데이터 없음"), None)

        df_sel = df[df["mold_code"].astype(str) == selected_code]
        if df_sel.empty:
            return _state("no_data", ALERT_STATUS_TEXT.get("no_data", "데이터 없음"), "선택 코드 데이터 없음")

        # 1) 외부 최신 예측 함수
        if callable(predict_latest_for_code):
            try:
                pred = predict_latest_for_code(selected_code)
                if pred is not None:
                    if isinstance(pred, dict):
                        if "is_defect" in pred:
                            is_def = bool(pred["is_defect"])
                        elif "prob" in pred:
                            is_def = float(pred["prob"]) >= float(proba_thresh)
                        else:
                            is_def = False
                    elif isinstance(pred, (int, np.integer, float, np.floating)):
                        is_def = float(pred) >= float(proba_thresh)
                    else:
                        is_def = bool(pred)
                    return _state("critical" if is_def else "normal",
                                  "CRITICAL (불량 예측)" if is_def else "정상",
                                  "불량으로 예측됨" if is_def else "정상 동작 중")
            except Exception as e:
                print(f"[defect_state] predict_latest_for_code 예외: {e}")

        # 2) 마지막 행을 모델에 넣어 예측
        row = df_sel.iloc[-1]
        if callable(predict_row):
            try:
                pr = predict_row(row)
                if isinstance(pr, dict):
                    if "is_defect" in pr:
                        is_def = bool(pr["is_defect"])
                    elif "prob" in pr:
                        is_def = float(pr["prob"]) >= float(proba_thresh)
                    else:
                        is_def = False
                elif isinstance(pr, (int, np.integer, float, np.floating)):
                    is_def = float(pr) >= float(proba_thresh)
                else:
                    is_def = bool(pr)
                return _state("critical" if is_def else "normal",
                              "CRITICAL (불량 예측)" if is_def else "정상",
                              "불량으로 예측됨" if is_def else "정상 동작 중")
            except Exception as e:
                print(f"[defect_state] predict_row 예외: {e}")

        # 3) 칼럼 기반 보조
        if pred_col and pred_col in row.index:
            pv = pd.to_numeric(row[pred_col], errors="coerce")
            if np.isfinite(pv):
                is_def = bool(int(pv)) if pv in (0, 1) else bool(float(pv) >= float(proba_thresh))
                return _state("critical" if is_def else "normal",
                              "CRITICAL (불량 예측)" if is_def else "정상",
                              "불량으로 예측됨" if is_def else "정상 동작 중")

        if proba_col and proba_col in row.index:
            prob = pd.to_numeric(row[proba_col], errors="coerce")
            if np.isfinite(prob):
                is_def = bool(float(prob) >= float(proba_thresh))
                return _state("critical" if is_def else "normal",
                              "CRITICAL (불량 예측)" if is_def else "정상",
                              "불량으로 예측됨" if is_def else "정상 동작 중")

        if "passorfail" in row.index:
            pf = pd.to_numeric(row["passorfail"], errors="coerce")
            if np.isfinite(pf):
                is_def = bool(int(pf) == 1)
                return _state("critical" if is_def else "normal",
                              "CRITICAL (불량 예측)" if is_def else "정상",
                              "불량(학습라벨)" if is_def else "정상 동작 중")

        # 모델 신호가 없으면 정상(초록)
        return _state("normal", "정상", "선택 코드 데이터 있음(모델 신호 없음)")

    # 불량 표시 UI들
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

    # 툴팁 빌더 (이상치용)
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

    # 오버레이 각 변수 출력 등록
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

    # 차트
    def create_single_chart(df, mold_code, variable, statuses=None, defect_info=None):
        # 💡 렌더 직전 OTF 폰트 재적용(워커/캐시 이슈 방지)
        setup_korean_font_otf_only()

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

        if df is None or df.empty:
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
                return empty(f'{"모든 Mold Code" if mold_code_str == "all" else f"Mold Code {mold_code}"} - {label}',
                             f'데이터 없음\n존재: {", ".join(codes)}', label, '#e74c3c')

            if variable not in filtered.columns:
                return empty(f'Mold Code {mold_code} - {label}', f'"{label}" 없음', label, '#e74c3c')

            if 'registration_time' in filtered.columns:
                filtered = filtered.sort_values('registration_time')
                plot_df = filtered[['registration_time', variable]].copy()
                plot_df[variable] = pd.to_numeric(plot_df[variable], errors='coerce')
                plot_df = plot_df.dropna().reset_index(drop=True).tail(10)
            else:
                plot_df = filtered[[variable]].copy()
                plot_df[variable] = pd.to_numeric(plot_df[variable], errors='coerce')
                plot_df = plot_df.dropna().reset_index(drop=True).tail(10)

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
            ax.set_title(f'{"모든 Mold Code" if mold_code_str == "all" else f"Mold Code {mold_code}"} - {label}',
                         fontsize=14, fontweight='bold', pad=20)
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
            statuses = anomaly_summary()   # 이상치(기존)
            defect_info = defect_state()   # 모델 불량 경보
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
        if df is None or df.empty:
            return render.DataGrid(
                pd.DataFrame({"메시지": ["데이터를 불러오는 중..."] + [""] * 9}),
                width="100%", filters=False, row_selection_mode="none"
            )
        result = df.tail(10).drop(columns=['line', 'name', 'mold_name'], errors='ignore').copy()
        if len(result) < 10:
            pad = pd.DataFrame([[None] * len(result.columns)] * (10 - len(result)), columns=result.columns)
            result = pd.concat([result, pad], ignore_index=True)
        return render.DataGrid(result, width="100%", filters=False, row_selection_mode="none")
