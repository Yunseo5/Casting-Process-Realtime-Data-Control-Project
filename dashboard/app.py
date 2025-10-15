# app.py
from shiny import App, ui

# -------------------------------------------------------------
# 1️⃣ 탭 모듈 import
# -------------------------------------------------------------
# 각 모듈 파일 내에 ui와 server 함수가 정의되어 있다고 가정
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
# 2️⃣ 전체 UI 구성 (네비게이션 탭)
# -------------------------------------------------------------
app_ui = ui.page_navbar(
    ui.nav_panel("현장 운영 담당자", operation_ui),
    ui.nav_panel("품질관리팀", qc_ui),
    ui.nav_panel("데이터 분석가", ai_ui),
    title="Casting Process Real-time Data Control Dashboard",
)

# -------------------------------------------------------------
# 3️⃣ 서버 로직 통합
# -------------------------------------------------------------
def server(input, output, session):
    operation_server(input, output, session)
    qc_server(input, output, session)
    ai_server(input, output, session)

# -------------------------------------------------------------
# 4️⃣ 앱 실행
# -------------------------------------------------------------
app = App(app_ui, server)
