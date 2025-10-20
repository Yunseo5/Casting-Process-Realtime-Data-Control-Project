# app.py
from pathlib import Path
from shiny import App, render, ui, reactive
from shared import RealTimeStreamer
import pandas as pd

from modules.tab_target_operation_manager import tab_ui as operation_ui, tab_server as operation_server
from modules.tab_target_qc_team import tab_ui as qc_ui, tab_server as qc_server
from modules.tab_target_ai_engineer import tab_ui as ai_ui, tab_server as ai_server
from modules.tab_target_log_accumulation import tab_ui as log_ui, tab_server as log_server

# -------------------------------------------------------------
# 탭 정의
# -------------------------------------------------------------
TAB_DEFINITIONS = [
    {"id": "operation", "label": "현장 운영 담당자", "icon": "fa-solid fa-gears", "content": operation_ui},
    {"id": "log", "label": "로그 누적", "icon": "fa-solid fa-database", "content": log_ui},
    {"id": "qc", "label": "품질관리팀", "icon": "fa-solid fa-clipboard-check", "content": qc_ui},
    {"id": "ai", "label": "데이터 분석가", "icon": "fa-solid fa-chart-line", "content": ai_ui},
]

TAB_CONTENT = {tab["id"]: tab["content"] for tab in TAB_DEFINITIONS}
DEFAULT_TAB = TAB_DEFINITIONS[0]["id"]

# -------------------------------------------------------------
# 스타일 및 스크립트
# -------------------------------------------------------------
app_assets = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }
body { background: #383636; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
#shiny-app-container { height: 100vh; display: flex; flex-direction: column; }

.outer-container {
    background: #000000; border-radius: 32px; padding: 16px; margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); height: calc(100vh - 40px);
    display: flex; flex-direction: column; position: relative;
}
.inner-container { border-radius: 24px; overflow: hidden; flex: 1; display: flex; flex-direction: column; }

/* 사이드바 스크롤 차단 */
.bslib-sidebar-layout, .bslib-sidebar-layout > aside, .bslib-sidebar-layout > aside *, #sidebar-nav {
    overflow: hidden !important;
}
.bslib-sidebar-layout > aside::-webkit-scrollbar { display: none !important; }
.bslib-sidebar-layout > aside {
    -ms-overflow-style: none !important; scrollbar-width: none !important;
    overscroll-behavior: none !important; touch-action: none !important;
}

/* 기본 collapse 버튼 제거 */
.bslib-sidebar-layout > .collapse-toggle, .bslib-sidebar-layout .collapse-toggle,
.bslib-sidebar-layout aside > .collapse-toggle, button.collapse-toggle,
.bslib-sidebar-layout > aside > button.collapse-toggle,
.bslib-sidebar-layout aside button[class*="collapse"],
aside > button:first-child, .sidebar > button:first-child {
    display: none !important; visibility: hidden !important; opacity: 0 !important;
    pointer-events: none !important; width: 0 !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important; position: absolute !important; left: -9999px !important;
}

/* ✅ 토글 버튼 - 완전히 분리된 고정 위치 */
#sidebar-toggle {
    position: fixed;
    top: 50%;
    left: 290px;
    transform: translateY(-50%);
    width: 42px;
    height: 42px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    z-index: 2000;
}

#sidebar-toggle:hover {
    background: rgba(74, 144, 226, 0.85);
    border-color: rgba(74, 144, 226, 0.9);
}

#sidebar-toggle i {
    font-size: 18px;
    transition: transform 0.3s ease;
}

body.sidebar-collapsed #sidebar-toggle {
    left: 88px;
    background: rgba(74, 144, 226, 0.9);
    border-color: rgba(74, 144, 226, 1);
}

body.sidebar-collapsed #sidebar-toggle i {
    transform: rotate(180deg);
}

/* 사이드바 레이아웃 */
.bslib-sidebar-layout {
    transition: grid-template-columns 0.3s ease; height: 100%; background: transparent !important;
}
.bslib-sidebar-layout > aside {
    background: #2A2D30 !important; border: none !important; padding: 32px 20px !important;
    display: flex !important; flex-direction: column; gap: 24px;
    transition: transform 0.3s ease;
}
body.sidebar-collapsed .bslib-sidebar-layout { grid-template-columns: 68px 1fr !important; }
body.sidebar-collapsed .bslib-sidebar-layout > aside {
    transform: translateX(calc(-100% + 68px)); padding: 24px 12px !important;
}
body.sidebar-collapsed .sidebar-title, body.sidebar-collapsed #sidebar-nav,
body.sidebar-collapsed .sidebar-controls { opacity: 0; pointer-events: none; }

/* 사이드바 컴포넌트 */
.sidebar-shell { width: 100%; display: flex; flex-direction: column; height: 100%; }
.sidebar-title {
    display: flex; flex-direction: column; gap: 4px; color: #ffffff;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;
    transition: opacity 0.3s ease; margin-bottom: 24px;
}
.sidebar-title span:last-child { font-size: 12px; opacity: 0.7; letter-spacing: 0.2em; }

#sidebar-nav { display: flex; flex-direction: column; gap: 8px; flex: 1; transition: opacity 0.3s ease; }
.sidebar-nav-item {
    display: flex; align-items: center; gap: 12px; padding: 12px 16px;
    border-radius: 12px; color: #ecf0f1; font-weight: 600; font-size: 15px;
    cursor: pointer; transition: all 0.2s ease; flex-shrink: 0;
}
.sidebar-nav-item:hover { background: rgba(255, 255, 255, 0.08); transform: translateX(4px); }
.sidebar-nav-item.active { background: #4A90E2; color: #ffffff; box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25); }
.sidebar-nav-item i { width: 20px; text-align: center; }

/* 사이드바 컨트롤 */
.sidebar-controls {
    margin-top: auto; padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1); transition: opacity 0.3s ease;
}
.sidebar-control-buttons { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.sidebar-btn {
    padding: 10px; border-radius: 10px; border: none; font-weight: 600; font-size: 14px;
    cursor: pointer; transition: all 0.2s; display: flex; align-items: center;
    justify-content: center; gap: 8px;
}
.sidebar-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); }
.sidebar-btn-start { background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); color: #fff; }
.sidebar-btn-stop { background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%); color: #fff; }
.sidebar-btn-reset { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: #fff; }
.sidebar-status {
    background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 12px;
    color: #fff; font-size: 13px; text-align: center; font-weight: 600;
}

/* 메인 영역 */
.dashboard-page { height: 100%; display: flex; flex-direction: column; }
.bslib-sidebar-layout > div.main {
    background: #F3F4F5 !important; padding: 32px !important;
    display: flex; justify-content: center; overflow: auto;
}
.main-scroll-container { flex: 1; width: 100%; max-width: 1400px; overflow-y: auto; }
.stats-card {
    background: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
</style>

<script>
(function() {
    'use strict';
    
    function removeCollapseButtons() {
        const sidebar = document.querySelector('.bslib-sidebar-layout > aside');
        if (!sidebar) return;
        sidebar.querySelectorAll('button').forEach(btn => {
            if (!btn.classList.contains('sidebar-btn') && btn.id !== 'sidebar-toggle') {
                btn.remove();
            }
        });
        sidebar.style.overflow = 'hidden';
    }

    function initSidebar() {
        const nav = document.getElementById('sidebar-nav');
        const hidden = document.getElementById('active_tab');
        if (!nav || !hidden || !window.Shiny) return;

        function setActive(tabId, emit) {
            if (!tabId) return;
            nav.querySelectorAll('.sidebar-nav-item').forEach(el => {
                el.classList.toggle('active', el.dataset.tab === tabId);
            });
            hidden.value = tabId;
            if (emit) window.Shiny.setInputValue('active_tab', tabId, { priority: 'event' });
        }

        nav.querySelectorAll('.sidebar-nav-item').forEach(el => {
            if (el.dataset.bound) return;
            el.dataset.bound = 'true';
            el.addEventListener('click', () => setActive(el.dataset.tab, true));
        });

        const toggleBtn = document.getElementById('sidebar-toggle');
        if (toggleBtn && !toggleBtn.dataset.bound) {
            toggleBtn.dataset.bound = 'true';
            toggleBtn.addEventListener('click', () => {
                document.body.classList.toggle('sidebar-collapsed');
            });
        }

        setActive(hidden.value, false);

        if (window.Shiny.addCustomMessageHandler) {
            window.Shiny.addCustomMessageHandler('set-active-tab', msg => {
                if (msg?.id) setActive(msg.id, Boolean(msg.emit));
            });
        }
        removeCollapseButtons();
    }

    if (document.readyState !== 'loading') initSidebar();
    else document.addEventListener('DOMContentLoaded', initSidebar);

    document.addEventListener('shiny:connected', () => {
        initSidebar();
        setTimeout(removeCollapseButtons, 100);
    });

    const observer = new MutationObserver(removeCollapseButtons);
    if (document.body) observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

# -------------------------------------------------------------
# UI 컴포넌트 생성
# -------------------------------------------------------------
def _nav_item(tab):
    classes = ["sidebar-nav-item"]
    if tab["id"] == DEFAULT_TAB:
        classes.append("active")
    return ui.div(
        ui.tags.i(class_=tab["icon"]),
        ui.span(tab["label"]),
        class_=" ".join(classes),
        **{"data-tab": tab["id"]},
    )

sidebar = ui.sidebar(
    ui.div(ui.input_text("active_tab", None, value=DEFAULT_TAB), style="display:none;"),
    ui.div(ui.span("Casting Process"), ui.span("대시보드"), class_="sidebar-title"),
    ui.div(*[_nav_item(tab) for tab in TAB_DEFINITIONS], id="sidebar-nav"),
    ui.div(
        ui.div(
            ui.input_action_button("sidebar_start_btn", ui.HTML('<i class="fa-solid fa-play"></i> 시작'), class_="sidebar-btn sidebar-btn-start"),
            ui.input_action_button("sidebar_stop_btn", ui.HTML('<i class="fa-solid fa-pause"></i> 정지'), class_="sidebar-btn sidebar-btn-stop"),
            ui.input_action_button("sidebar_reset_btn", ui.HTML('<i class="fa-solid fa-rotate-right"></i> 리셋'), class_="sidebar-btn sidebar-btn-reset"),
            class_="sidebar-control-buttons",
        ),
        ui.div(ui.output_ui("sidebar_status_text"), class_="sidebar-status"),
        class_="sidebar-controls",
    ),
    class_="sidebar-shell",
)

# ✅ 토글 버튼을 outer-container에 직접 배치
app_ui = ui.page_fluid(
    ui.HTML(app_assets),
    ui.div(
        ui.tags.button(
            ui.tags.i(class_="fa-solid fa-chevron-left"),
            id="sidebar-toggle",
            type="button",
        ),
        ui.div(
            ui.page_sidebar(
                sidebar,
                ui.div(ui.output_ui("active_tab_content"), class_="main-scroll-container"),
                class_="dashboard-page", fillable=True,
            ),
            class_="inner-container",
        ),
        class_="outer-container",
    ),
)

# -------------------------------------------------------------
# 서버 로직
# -------------------------------------------------------------
def server(input, output, session):
    streamer = RealTimeStreamer()
    shared_df = reactive.Value(pd.DataFrame())
    streaming_active = reactive.Value(False)
    
    @reactive.effect
    def _update():
        reactive.invalidate_later(0.5)
        if streaming_active.get():
            shared_df.set(streamer.get_current_data())
    
    @reactive.effect
    @reactive.event(input.sidebar_start_btn)
    def _(): streamer.start_stream(); streaming_active.set(True)

    @reactive.effect
    @reactive.event(input.sidebar_stop_btn)
    def _(): streamer.stop_stream(); streaming_active.set(False)

    @reactive.effect
    @reactive.event(input.sidebar_reset_btn)
    def _(): streamer.reset_stream(); shared_df.set(pd.DataFrame()); streaming_active.set(False)
    
    @output
    @render.ui
    def sidebar_status_text():
        _ = shared_df.get()
        icon = '<i class="fa-solid fa-circle-play"></i>' if streaming_active.get() else '<i class="fa-solid fa-circle-pause"></i>'
        status = "진행 중" if streaming_active.get() else "정지"
        return ui.HTML(f'{icon} {status}<br>진행률: {streamer.progress():.1f}%')
    
    @render.ui
    def active_tab_content():
        return TAB_CONTENT.get(input.active_tab() or DEFAULT_TAB, TAB_CONTENT[DEFAULT_TAB])

    operation_server(input, output, session, streamer, shared_df, streaming_active)
    log_server(input, output, session, streamer, shared_df, streaming_active)
    qc_server(input, output, session)
    ai_server(input, output, session)
    
    session.on_ended(streamer.cleanup)

app = App(app_ui, server, static_assets=str(Path(__file__).parent / "data" / "png"))