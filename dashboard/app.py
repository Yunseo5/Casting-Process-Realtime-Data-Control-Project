# app.py
from pathlib import Path
from shiny import App, render, ui
from modules.tab_target_operation_manager import (
    tab_ui as operation_ui,
    tab_server as operation_server,
)
from modules.tab_target_qc_team import (
    tab_ui as qc_ui,
    tab_server as qc_server,
)
from modules.tab_target_ai_engineer import (
    tab_ui as ai_ui,
    tab_server as ai_server,
)

# -------------------------------------------------------------
# 탭 정의
# -------------------------------------------------------------
TAB_DEFINITIONS = [
    {
        "id": "operation",
        "label": "현장 운영 담당자",
        "icon": "fa-solid fa-gears",
        "content": operation_ui,
    },
    {
        "id": "qc",
        "label": "품질관리팀",
        "icon": "fa-solid fa-clipboard-check",
        "content": qc_ui,
    },
    {
        "id": "ai",
        "label": "데이터 분석가",
        "icon": "fa-solid fa-chart-line",
        "content": ai_ui,
    },
]

TAB_CONTENT = {tab["id"]: tab["content"] for tab in TAB_DEFINITIONS}
DEFAULT_TAB = TAB_DEFINITIONS[0]["id"]

# -------------------------------------------------------------
# 스타일 및 스크립트
# -------------------------------------------------------------
app_assets = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
}

body {
    background: #383636;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#shiny-app-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.outer-container {
    background: #000000;
    border-radius: 32px;
    padding: 16px;
    margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    height: calc(100vh - 40px);
    display: flex;
    flex-direction: column;
}

.inner-container {
    border-radius: 24px;
    overflow: hidden;
    flex: 1;
    display: flex;
    flex-direction: column;
}

/* ---------------------------------------------- */
/* 🔒 사이드바 스크롤바 및 스크롤 동작 완전 차단 */
/* ---------------------------------------------- */
.bslib-sidebar-layout,
.bslib-sidebar-layout > aside,
.bslib-sidebar-layout > aside *,
#sidebar-nav {
    overflow: hidden !important;
}

.bslib-sidebar-layout > aside::-webkit-scrollbar {
    display: none !important;
}

.bslib-sidebar-layout > aside {
    -ms-overflow-style: none !important;
    scrollbar-width: none !important;
    overscroll-behavior: none !important;
    touch-action: none !important;
}

/* 모든 collapse 관련 요소 완전히 제거 */
.bslib-sidebar-layout > .collapse-toggle,
.bslib-sidebar-layout .collapse-toggle,
.bslib-sidebar-layout aside > .collapse-toggle,
button.collapse-toggle,
.bslib-sidebar-layout > aside > button.collapse-toggle,
.bslib-sidebar-layout aside button[class*="collapse"],
aside > button:first-child,
.sidebar > button:first-child {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
    left: -9999px !important;
}

/* ---------------------------------------------- */
/* 일반 UI 스타일 */
/* ---------------------------------------------- */
.sidebar-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 24px;
}

.sidebar-toggle-button {
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
    z-index: 10;
    flex-shrink: 0;
}

.sidebar-toggle-button:hover {
    background: rgba(74, 144, 226, 0.85);
    border-color: rgba(74, 144, 226, 0.9);
}

.sidebar-toggle-button i {
    font-size: 18px;
    transition: transform 0.3s ease;
}

.sidebar-toggle-button.collapsed i {
    transform: rotate(180deg);
}

.bslib-sidebar-layout {
    transition: grid-template-columns 0.3s ease, margin 0.3s ease;
    height: 100%;
}

body.sidebar-collapsed .bslib-sidebar-layout {
    grid-template-columns: 68px 1fr !important;
}

body.sidebar-collapsed .bslib-sidebar-layout > aside {
    transform: translateX(calc(-100% + 68px));
    padding: 24px 12px !important;
}

body.sidebar-collapsed .sidebar-title,
body.sidebar-collapsed #sidebar-nav {
    opacity: 0;
    pointer-events: none;
}

body.sidebar-collapsed .sidebar-header {
    justify-content: flex-end;
}

body.sidebar-collapsed .sidebar-toggle-button {
    background: rgba(74, 144, 226, 0.9);
    border-color: rgba(74, 144, 226, 1);
}

.dashboard-page {
    height: 100%;
    display: flex;
    flex-direction: column;
}

.bslib-sidebar-layout {
    background: transparent !important;
}

.bslib-sidebar-layout > aside {
    background: #2A2D30 !important;
    border: none !important;
    padding: 32px 20px !important;
    display: flex !important;
    flex-direction: column;
    gap: 24px;
}

.sidebar-shell {
    width: 100%;
    display: flex;
    flex-direction: column;
    height: 100%;
}

.sidebar-title {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    text-align: left;
}

.sidebar-title span:last-child {
    font-size: 12px;
    opacity: 0.7;
    letter-spacing: 0.2em;
}

#sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    color: #ecf0f1;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
    transition: all 0.2s ease;
    flex-shrink: 0;
}

.sidebar-nav-item:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(4px);
}

.sidebar-nav-item.active {
    background: #4A90E2;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25);
}

.sidebar-nav-item i {
    width: 20px;
    text-align: center;
}

.bslib-sidebar-layout > div.main {
    background: #F3F4F5 !important;
    padding: 32px !important;
    display: flex;
    justify-content: center;
    overflow: auto;
}

.main-scroll-container {
    flex: 1;
    width: 100%;
    max-width: 1400px;
    overflow-y: auto;
}
</style>

<script>
(function() {
    function removeCollapseButtons() {
        const sidebar = document.querySelector('.bslib-sidebar-layout > aside');
        if (sidebar) {
            const buttons = sidebar.querySelectorAll('button');
            buttons.forEach(btn => {
                if (!btn.classList.contains('sidebar-toggle-button') && btn.id !== 'sidebar-toggle') {
                    btn.remove();
                }
            });
        }
        // 🔒 스크롤 완전 차단 (렌더링 후 다시 생기는 경우 방지)
        if (sidebar) {
            sidebar.style.overflow = 'hidden';
        }
    }

    function initSidebar() {
        const nav = document.getElementById('sidebar-nav');
        const hidden = document.getElementById('active_tab');
        if (!nav || !hidden || !window.Shiny) return;

        function setActive(tabId, emit) {
            if (!tabId) return;
            nav.querySelectorAll('.sidebar-nav-item').forEach((el) => {
                el.classList.toggle('active', el.dataset.tab === tabId);
            });
            hidden.value = tabId;
            if (emit) window.Shiny.setInputValue('active_tab', tabId, { priority: 'event' });
        }

        nav.querySelectorAll('.sidebar-nav-item').forEach((el) => {
            if (el.dataset.bound === 'true') return;
            el.dataset.bound = 'true';
            el.addEventListener('click', () => setActive(el.dataset.tab, true));
        });

        const layout = document.querySelector('.bslib-sidebar-layout');
        const toggleBtn = document.getElementById('sidebar-toggle');
        const collapsed = document.body.classList.contains('sidebar-collapsed');
        if (layout) layout.classList.toggle('collapsed', collapsed);
        if (toggleBtn) {
            toggleBtn.classList.toggle('collapsed', collapsed);
            if (!toggleBtn.dataset.bound) {
                toggleBtn.dataset.bound = 'true';
                toggleBtn.addEventListener('click', () => {
                    const next = !document.body.classList.contains('sidebar-collapsed');
                    document.body.classList.toggle('sidebar-collapsed', next);
                    if (layout) layout.classList.toggle('collapsed', next);
                    toggleBtn.classList.toggle('collapsed', next);
                });
            }
        }

        const initial = hidden.value;
        setActive(initial, false);

        if (window.Shiny.addCustomMessageHandler) {
            window.Shiny.addCustomMessageHandler('set-active-tab', (msg) => {
                if (msg && msg.id) setActive(msg.id, Boolean(msg.emit));
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

    const observer = new MutationObserver(() => {
        removeCollapseButtons();
    });
    if (document.body) observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


# -------------------------------------------------------------
# 사이드바 네비게이션 아이템 생성
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

SIDEBAR_NAV = ui.div(*(_nav_item(tab) for tab in TAB_DEFINITIONS), id="sidebar-nav")

# -------------------------------------------------------------
# 사이드바 구성
# -------------------------------------------------------------
sidebar = ui.sidebar(
    ui.div(ui.input_text("active_tab", None, value=DEFAULT_TAB), style="display:none;"),
    ui.div(
        ui.div(
            ui.span("Casting Process"),
            ui.span("대시보드"),
            class_="sidebar-title",
        ),
        ui.tags.button(
            ui.tags.i(class_="fa-solid fa-chevron-left"),
            id="sidebar-toggle",
            class_="sidebar-toggle-button",
            type="button",
        ),
        class_="sidebar-header",
    ),
    SIDEBAR_NAV,
    class_="sidebar-shell",
)

# -------------------------------------------------------------
# 전체 UI 구성
# -------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.HTML(app_assets),
    ui.div(
        ui.div(
            ui.page_sidebar(
                sidebar,
                ui.div(
                    ui.output_ui("active_tab_content"),
                    class_="main-scroll-container",
                ),
                class_="dashboard-page",
                fillable=True,
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
    @render.ui
    def active_tab_content():
        tab_id = input.active_tab() or DEFAULT_TAB
        return TAB_CONTENT.get(tab_id, TAB_CONTENT[DEFAULT_TAB])

    operation_server(input, output, session)
    qc_server(input, output, session)
    ai_server(input, output, session)

# -------------------------------------------------------------
# 앱 실행
# -------------------------------------------------------------
static_path = Path(__file__).parent / "data" / "png"
app = App(app_ui, server, static_assets=str(static_path))