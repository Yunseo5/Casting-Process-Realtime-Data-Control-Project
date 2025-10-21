from pathlib import Path
from shiny import App, render, ui, reactive
from shared import RealTimeStreamer
import pandas as pd

from modules.tab_target_operation_manager import tab_ui as operation_ui, tab_server as operation_server
from modules.tab_target_qc_team import tab_ui as qc_ui, tab_server as qc_server
from modules.tab_target_ai_engineer import tab_ui as ai_ui, tab_server as ai_server
from modules.tab_target_log_accumulation import tab_ui as log_ui, tab_server as log_server

## -------------------------------------------------------------
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
.inner-container { 
    border-radius: 24px; overflow: hidden; flex: 1; 
    display: flex;
}

/* ✅ 사이드바 컨테이너 */
.sidebar-container {
    background: #2A2D30;
    transition: width 0.3s ease;
    width: 240px;
    overflow: visible;
    display: flex;
    flex-direction: column;
}

.sidebar-container.collapsed {
    width: 70px;
}

/* ✅ 사이드바 내부 */
.sidebar-inner {
    padding: 16px 0px 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    height: 100%;
    overflow: visible;
}

.sidebar-container.collapsed .sidebar-inner {
    padding: 20px 10px;
}

/* ✅ 토글 버튼 - 사이드바 내부 상단에 배치 */
#sidebar-toggle-btn {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
}

#sidebar-toggle-btn:hover {
    background: linear-gradient(135deg, rgba(74, 144, 226, 0.25) 0%, rgba(74, 144, 226, 0.15) 100%);
    border-color: rgba(74, 144, 226, 0.5);
    box-shadow: 0 4px 16px rgba(74, 144, 226, 0.3);
    transform: scale(1.08);
}

#sidebar-toggle-btn i {
    font-size: 14px;
    transition: transform 0.3s ease;
}

.sidebar-container.collapsed #sidebar-toggle-btn i {
    transform: rotate(180deg);
}

/* ✅ 사이드바 컨텐츠 숨김 처리 */
.sidebar-content {
    opacity: 1;
    transition: opacity 0.3s ease;
    display: flex;
    flex-direction: column;
    gap: 24px;
    flex: 1;
}

.sidebar-container.collapsed .sidebar-content {
    opacity: 0;
    pointer-events: none;
}

/* 사이드바 컴포넌트 */
.sidebar-title {
    display: flex; flex-direction: column; gap: 4px; color: #ffffff;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;
    margin-bottom: 12px;
    margin-top: -8px;
    padding-left: 8px;
}
.sidebar-title span:last-child { font-size: 12px; opacity: 0.7; letter-spacing: 0.2em; }

#sidebar-nav { display: flex; flex-direction: column; gap: 8px; flex: 1; }
.sidebar-nav-item {
    display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-radius: 12px 0 0 12px;
    color: #ecf0f1; font-weight: 600; font-size: 14px;
    cursor: pointer; transition: all 0.2s ease; flex-shrink: 0;
    white-space: nowrap;
    width: calc(100% + 32px);
    max-width: none;
    position: relative;
}
.sidebar-nav-item:hover { background: rgba(255, 255, 255, 0.08); transform: translateX(4px); }
.sidebar-nav-item.active { 
    background: #F3F4F5; 
    color: #2A2D30; 
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    width: calc(100% + 32px);
}
.sidebar-nav-item i { width: 18px; text-align: center; font-size: 16px; flex-shrink: 0; }
.sidebar-nav-item span { overflow: hidden; text-overflow: ellipsis; }

/* 사이드바 컨트롤 */
.sidebar-controls {
    margin-top: auto; padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
}

.sidebar-buttons-row {
    display: flex;
    gap: 12px;
    justify-content: flex-start;
    align-items: center;
}

/* ✅ FAB 스타일 버튼 - 3D 원통 느낌 */
.fab-sidebar-button {
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    box-shadow: 
        0 6px 12px rgba(0, 0, 0, 0.4),
        0 3px 6px rgba(0, 0, 0, 0.3),
        inset 0 -3px 6px rgba(0, 0, 0, 0.3),
        inset 0 2px 3px rgba(255, 255, 255, 0.3);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
    transition: all 0.2s ease;
    position: relative;
    overflow: visible;
    padding: 0 !important;
    transform: translateY(0);
}

.fab-sidebar-button::after {
    content: '';
    position: absolute;
    top: -2px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 30%;
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.4), transparent);
    border-radius: 50% 50% 50% 50% / 80% 80% 20% 20%;
    pointer-events: none;
}

.fab-sidebar-button:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 10px 20px rgba(0, 0, 0, 0.5),
        0 6px 10px rgba(0, 0, 0, 0.4),
        inset 0 -4px 8px rgba(0, 0, 0, 0.3),
        inset 0 2px 4px rgba(255, 255, 255, 0.3);
}

.fab-sidebar-button:active {
    transform: translateY(2px);
    box-shadow: 
        0 4px 8px rgba(0, 0, 0, 0.4),
        0 2px 4px rgba(0, 0, 0, 0.3),
        inset 0 -2px 4px rgba(0, 0, 0, 0.3),
        inset 0 1px 2px rgba(255, 255, 255, 0.3);
}

.fab-icon {
    font-size: 20px;
    z-index: 1;
}

.fab-text {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    z-index: 1;
}

/* FAB 색상 - 시작 (빨간색) */
.fab-start {
    background: linear-gradient(135deg, #8E0000 0%, #b30000 100%);
    color: #fff;
}

.fab-start:hover {
    background: linear-gradient(135deg, #b30000 0%, #d60000 100%);
}

/* FAB 색상 - 정지 (초록색) */
.fab-stop {
    background: linear-gradient(135deg, #556B35 0%, #6b8641 100%);
    color: #fff;
}

.fab-stop:hover {
    background: linear-gradient(135deg, #6b8641 0%, #7a9850 100%);
}

.sidebar-reset-container {
    display: flex;
    justify-content: center;
}

.sidebar-btn-reset { 
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;
    border-radius: 50%;
    background: linear-gradient(135deg, #EDC313 0%, #f0d948 100%); 
    color: #333;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 
        0 6px 12px rgba(0, 0, 0, 0.4),
        0 3px 6px rgba(0, 0, 0, 0.3),
        inset 0 -3px 6px rgba(0, 0, 0, 0.2),
        inset 0 2px 3px rgba(255, 255, 255, 0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
    position: relative;
    overflow: visible;
    padding: 0 !important;
    transform: translateY(0);
}

.sidebar-btn-reset::after {
    content: '';
    position: absolute;
    top: -2px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 30%;
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.6), transparent);
    border-radius: 50% 50% 50% 50% / 80% 80% 20% 20%;
    pointer-events: none;
}

.sidebar-btn-reset:hover {
    background: linear-gradient(135deg, #f0d948 0%, #f5e574 100%);
    transform: translateY(-2px);
    box-shadow: 
        0 10px 20px rgba(0, 0, 0, 0.5),
        0 6px 10px rgba(0, 0, 0, 0.4),
        inset 0 -4px 8px rgba(0, 0, 0, 0.2),
        inset 0 2px 4px rgba(255, 255, 255, 0.5);
}

.sidebar-btn-reset:active {
    transform: translateY(2px);
    box-shadow: 
        0 4px 8px rgba(0, 0, 0, 0.4),
        0 2px 4px rgba(0, 0, 0, 0.3),
        inset 0 -2px 4px rgba(0, 0, 0, 0.2),
        inset 0 1px 2px rgba(255, 255, 255, 0.5);
}

.reset-icon {
    font-size: 20px;
    z-index: 1;
}

.reset-text {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    z-index: 1;
}

.sidebar-status {
    /* 스타일 제거 - 인라인 스타일 사용 */
}

.status-icon {
    font-size: 16px;
    animation: pulse 2s ease-in-out infinite;
}

.status-text {
    flex: 1;
    letter-spacing: 0.3px;
}

/* 진행 중일 때 아이콘 애니메이션 */
@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.7;
        transform: scale(1.1);
    }
}

/* 상태별 색상 */
.status-running {
    border-left: 3px solid #4CAF50;
}

.status-stopped {
    border-left: 3px solid #9E9E9E;
}

/* ✅ 메인 영역 */
.main-content {
    flex: 1;
    background: #F3F4F5;
    padding: 32px;
    overflow-y: auto;
    display: flex;
    justify-content: center;
    position: relative;
}

.main-scroll-container { 
    width: 100%; 
    max-width: 1400px; 
}

.stats-card {
    background: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
</style>

<script>
(function() {
    'use strict';
    
    function initSidebar() {
        const nav = document.getElementById('sidebar-nav');
        const hidden = document.getElementById('active_tab');
        const sidebarContainer = document.querySelector('.sidebar-container');
        const toggleBtn = document.getElementById('sidebar-toggle-btn');
        
        if (!nav || !hidden || !window.Shiny) return;

        // ✅ 탭 전환 로직
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

        // ✅ 토글 버튼 클릭 이벤트
        if (toggleBtn && !toggleBtn.dataset.bound) {
            toggleBtn.dataset.bound = 'true';
            toggleBtn.addEventListener('click', () => {
                sidebarContainer.classList.toggle('collapsed');
            });
        }

        setActive(hidden.value, false);

        if (window.Shiny.addCustomMessageHandler) {
            window.Shiny.addCustomMessageHandler('set-active-tab', msg => {
                if (msg?.id) setActive(msg.id, Boolean(msg.emit));
            });
        }
    }

    if (document.readyState !== 'loading') initSidebar();
    else document.addEventListener('DOMContentLoaded', initSidebar);

    document.addEventListener('shiny:connected', () => {
        initSidebar();
    });
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

# ✅ 사이드바 구조
def create_sidebar():
    return ui.div(
        ui.div(
            # 토글 버튼
            ui.tags.button(
                ui.tags.i(class_="fa-solid fa-bars"),
                id="sidebar-toggle-btn",
                type="button",
            ),
            # 사이드바 컨텐츠
            ui.div(
                ui.div(ui.span("CASTING"), ui.span("PROCESS"), ui.span("대시보드"), class_="sidebar-title"),
                ui.div(*[_nav_item(tab) for tab in TAB_DEFINITIONS], id="sidebar-nav"),
                ui.div(
                    ui.div(
                        ui.output_ui("fab_button"),
                        ui.input_action_button(
                            "sidebar_reset_btn", 
                            ui.HTML('<div class="reset-icon"><i class="fa-solid fa-rotate-right"></i></div><div class="reset-text">리셋</div>'), 
                            class_="sidebar-btn-reset"
                        ),
                        class_="sidebar-buttons-row",
                    ),
                    ui.output_ui("sidebar_status_text"),
                    class_="sidebar-controls",
                ),
                class_="sidebar-content",
            ),
            class_="sidebar-inner",
        ),
        class_="sidebar-container",
    )

app_ui = ui.page_fluid(
    ui.HTML(app_assets),
    ui.div(ui.input_text("active_tab", None, value=DEFAULT_TAB), style="display:none;"),
    ui.div(
        ui.div(
            create_sidebar(),
            ui.div(
                ui.div(ui.output_ui("active_tab_content"), class_="main-scroll-container"),
                class_="main-content",
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
            df = streamer.get_current_data()
            shared_df.set(df)
    
    @reactive.effect
    @reactive.event(input.fab_toggle_btn)
    def _toggle_stream(): 
        if streaming_active.get():
            streamer.stop_stream()
            streaming_active.set(False)
        else:
            streamer.start_stream()
            streaming_active.set(True)
    
    @output
    @render.ui
    def fab_button():
        is_streaming = streaming_active.get()
        if is_streaming:
            # 정지 상태 - 초록색
            btn_class = "fab-sidebar-button fab-stop"
            icon_class = "fa-solid fa-pause"
            text = "정지"
        else:
            # 시작 상태 - 빨간색
            btn_class = "fab-sidebar-button fab-start"
            icon_class = "fa-solid fa-play"
            text = "시작"
        
        return ui.tags.button(
            ui.div(ui.tags.i(class_=icon_class), class_="fab-icon"),
            ui.div(text, class_="fab-text"),
            id="fab_toggle_btn",
            class_=btn_class,
            onclick="Shiny.setInputValue('fab_toggle_btn', Math.random())"
        )

    @reactive.effect
    @reactive.event(input.sidebar_reset_btn)
    def _(): 
        streamer.reset_stream()
        shared_df.set(pd.DataFrame())
        streaming_active.set(False)
    
    @output
    @render.ui
    def sidebar_status_text():
        _ = shared_df.get()
        is_running = streaming_active.get()
        
        if is_running:
            icon = '<i class="fa-solid fa-circle-play"></i>'
            status = "진행 중"
            color = "#4CAF50"
        else:
            icon = '<i class="fa-solid fa-circle-pause"></i>'
            status = "정지"
            color = "#9E9E9E"
        
        status_style = f"""
            display: flex;
            align-items: center;
            gap: 8px;
            color: {color};
            font-size: 13px;
            font-weight: 600;
            white-space: nowrap;
        """
        
        return ui.div(
            ui.HTML(f'<span style="font-size: 14px;">{icon}</span>'),
            ui.HTML(f'<span style="letter-spacing: 0.3px;">{status}</span>'),
            style=status_style
        )
    
    @render.ui
    def active_tab_content():
        return TAB_CONTENT.get(input.active_tab() or DEFAULT_TAB, TAB_CONTENT[DEFAULT_TAB])

    # ✅ 탭 서버 호출 (간소화)
    operation_server(input, output, session, streamer, shared_df, streaming_active)
    log_server(input, output, session, streamer, shared_df, streaming_active)
    qc_server(input, output, session)
    ai_server(input, output, session)
    
    session.on_ended(streamer.cleanup)

app = App(app_ui, server, static_assets=str(Path(__file__).parent / "data" / "png"))