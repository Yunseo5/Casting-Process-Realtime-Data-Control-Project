# modules/tab_target_qc_team.py
from shiny import ui, render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 데이터 로드 및 필터링 함수
def load_and_filter_data(date_start=None, date_end=None):
    # modules/tab_target_qc_team.py -> modules -> dashboard -> Project root -> data
    data_path = Path(__file__).parent.parent.parent / "data" / "train.csv"
    df = pd.read_csv(data_path)
    
    # registration_time 컬럼을 datetime으로 변환
    df['registration_time'] = pd.to_datetime(df['registration_time'])
    
    # 날짜 필터링 (registration_time 기준)
    if date_start is not None:
        df = df[df['registration_time'].dt.date >= pd.to_datetime(date_start).date()]
    if date_end is not None:
        df = df[df['registration_time'].dt.date <= pd.to_datetime(date_end).date()]
    
    return df

# P 관리도 계산 함수 (날짜 기반 - registration_time 사용)
def calculate_p_chart_by_date(df):
    # registration_time에서 날짜만 추출
    df['date_only'] = df['registration_time'].dt.date
    
    # 날짜별로 그룹화하여 불량률 계산
    date_stats = df.groupby('date_only').agg(
        defects=('passorfail', 'sum'),
        total=('passorfail', 'count')
    ).reset_index()
    
    # 불량률 계산
    date_stats['p'] = date_stats['defects'] / date_stats['total']
    
    # 전체 불량률 (중심선, CL)
    p_bar = date_stats['defects'].sum() / date_stats['total'].sum()
    
    # 평균 샘플 크기
    n_bar = date_stats['total'].mean()
    
    # 관리한계선 계산
    sigma = np.sqrt(p_bar * (1 - p_bar) / n_bar)
    
    UCL = p_bar + 3 * sigma
    LCL = p_bar - 3 * sigma
    LCL = max(0, LCL)
    
    return date_stats, p_bar, UCL, LCL
def calculate_p_chart(df, subgroup_size=5):
    # passorfail 컬럼 확인
    if 'passorfail' not in df.columns:
        raise ValueError("passorfail 컬럼이 없습니다.")
    
    # 데이터를 서브그룹으로 묶기
    n_subgroups = len(df) // subgroup_size
    df_subgroups = df.head(n_subgroups * subgroup_size).copy()
    
    # 서브그룹 인덱스 추가
    df_subgroups['subgroup'] = np.repeat(range(n_subgroups), subgroup_size)
    
    # 각 서브그룹의 불량 개수 계산
    subgroup_stats = df_subgroups.groupby('subgroup').agg(
        defects=('passorfail', 'sum'),  # 불량 개수 (passorfail=1의 합)
        total=('passorfail', 'count')    # 서브그룹 크기
    ).reset_index()
    
    # 불량률 계산
    subgroup_stats['p'] = subgroup_stats['defects'] / subgroup_stats['total']
    
    # 전체 불량률 (중심선, CL)
    p_bar = subgroup_stats['defects'].sum() / subgroup_stats['total'].sum()
    
    # 관리한계선 계산
    n = subgroup_size
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    
    UCL = p_bar + 3 * sigma
    LCL = p_bar - 3 * sigma
    LCL = max(0, LCL)  # LCL은 0 이상이어야 함
    
    return subgroup_stats, p_bar, UCL, LCL

# Xbar-R 관리도 계산 함수 (날짜 기반 - registration_time 사용)
def calculate_xbar_r_chart_by_date(df, variable):
    # registration_time에서 날짜만 추출
    df['date_only'] = df['registration_time'].dt.date
    
    # 날짜별로 그룹화하여 평균과 범위 계산
    date_stats = df.groupby('date_only')[variable].agg([
        ('mean', 'mean'),
        ('range', lambda x: x.max() - x.min())
    ]).reset_index()
    
    # 전체 평균 (Xbar_bar)
    xbar_bar = date_stats['mean'].mean()
    
    # 전체 범위 평균 (R_bar)
    r_bar = date_stats['range'].mean()
    
    # 평균 샘플 크기 계산
    n_bar = df.groupby('date_only').size().mean()
    
    # 관리한계선 계산 상수 (평균 샘플 크기 사용)
    control_chart_constants = {
        2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
        3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
        4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
        5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
        6: {'A2': 0.483, 'D3': 0, 'D4': 2.004},
        7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
        8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864},
        9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
        10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}
    }
    
    n_rounded = int(round(n_bar))
    n_rounded = max(2, min(10, n_rounded))
    constants = control_chart_constants.get(n_rounded, control_chart_constants[5])
    
    A2 = constants['A2']
    D3 = constants['D3']
    D4 = constants['D4']
    
    # Xbar 차트 관리한계선
    UCL_xbar = xbar_bar + A2 * r_bar
    LCL_xbar = xbar_bar - A2 * r_bar
    
    # R 차트 관리한계선
    UCL_r = D4 * r_bar
    LCL_r = D3 * r_bar
    
    return date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r

# Xbar-R 관리도 계산 함수
def calculate_xbar_r_chart(df, variable, subgroup_size=5):
    # 변수 확인
    if variable not in df.columns:
        raise ValueError(f"{variable} 컬럼이 없습니다.")
    
    # 데이터를 서브그룹으로 묶기
    n_subgroups = len(df) // subgroup_size
    df_subgroups = df.head(n_subgroups * subgroup_size).copy()
    
    # 서브그룹 인덱스 추가
    df_subgroups['subgroup'] = np.repeat(range(n_subgroups), subgroup_size)
    
    # 각 서브그룹의 평균과 범위 계산
    subgroup_stats = df_subgroups.groupby('subgroup')[variable].agg([
        ('mean', 'mean'),
        ('range', lambda x: x.max() - x.min())
    ]).reset_index()
    
    # 전체 평균 (Xbar_bar)
    xbar_bar = subgroup_stats['mean'].mean()
    
    # 전체 범위 평균 (R_bar)
    r_bar = subgroup_stats['range'].mean()
    
    # 관리한계선 계산 상수 (서브그룹 크기에 따라 달라짐)
    # A2, D3, D4 상수표
    control_chart_constants = {
        2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
        3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
        4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
        5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
        6: {'A2': 0.483, 'D3': 0, 'D4': 2.004},
        7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
        8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864},
        9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
        10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}
    }
    
    # n=1의 경우 특별 처리 (이동 범위 사용)
    if subgroup_size == 1:
        # 이동 범위 계산
        moving_ranges = df_subgroups[variable].diff().abs()
        r_bar = moving_ranges.mean()
        constants = {'A2': 2.660, 'D3': 0, 'D4': 3.267}
    else:
        constants = control_chart_constants.get(subgroup_size, control_chart_constants[5])
    
    A2 = constants['A2']
    D3 = constants['D3']
    D4 = constants['D4']
    
    # Xbar 차트 관리한계선
    UCL_xbar = xbar_bar + A2 * r_bar
    LCL_xbar = xbar_bar - A2 * r_bar
    
    # R 차트 관리한계선
    UCL_r = D4 * r_bar
    LCL_r = D3 * r_bar
    
    return subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r

# 탭별 UI
tab_ui = ui.page_fluid(
    ui.h2("Quality Control Team - Control Charts", style="margin-bottom: 30px;"),
    
    # 필터 설정 (접을 수 있는 Accordion)
    ui.accordion(
        ui.accordion_panel(
            "Filter Settings",
            # 모드 선택
            ui.card(
                ui.card_header("Analysis Mode"),
                ui.input_radio_buttons(
                    "analysis_mode",
                    None,
                    choices={
                        "subgroup": "Subgroup-based Analysis",
                        "date": "Date-based Analysis"
                    },
                    selected="subgroup",
                    inline=True
                ),
            ),
            
            # 서브그룹 모드 설정
            ui.panel_conditional(
                "input.analysis_mode === 'subgroup'",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Subgroup Size (Fixed: n=5)"),
                        ui.HTML("<p style='text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;'>n = 5</p>"),
                    ),
                    ui.card(
                        ui.card_header("Subgroup Range Selection"),
                        ui.input_numeric("subgroup_start", "Start Subgroup", value=0, min=0),
                        ui.input_numeric("subgroup_end", "End Subgroup", value=100, min=1),
                    ),
                    col_widths=[6, 6]
                ),
            ),
            
            # 날짜 모드 설정
            ui.panel_conditional(
                "input.analysis_mode === 'date'",
                ui.card(
                    ui.card_header("Date Range Selection"),
                    ui.input_date_range(
                        "date_range",
                        "Select Date Range (2019-01-02 ~ 2019-03-12)",
                        start="2019-01-02",
                        end="2019-03-12",
                        min="2019-01-02",
                        max="2019-03-12",
                        width="100%"
                    ),
                ),
            ),
            
            ui.input_action_button("apply_filter", "Apply Filter", class_="btn-primary", style="margin-top: 15px;"),
        ),
        open=False
    ),
    
    # P 관리도
    ui.card(
        ui.card_header("P Control Chart (Defect Rate)"),
        ui.output_plot("plot_p_chart", height="600px"),
    ),
    
    # 변수 선택 (접을 수 있는 Accordion)
    ui.accordion(
        ui.accordion_panel(
            "Variable Selection for Xbar-R Control Chart",
            ui.tags.style("""
            #xbar_variable {
                display: grid !important;
                grid-template-columns: repeat(3, 1fr) !important;
                gap: 8px 15px !important;
                max-width: 100% !important;
            }
            #xbar_variable .radio {
                margin: 0 !important;
            }
            #xbar_variable label {
                margin: 0 !important;
                display: flex !important;
                align-items: center !important;
                white-space: nowrap !important;
            }
            #xbar_variable input[type="radio"] {
                margin-right: 5px !important;
                flex-shrink: 0 !important;
            }
            """),
            ui.input_radio_buttons(
                "xbar_variable",
                None,
                choices={
                    "molten_temp": "Molten Temperature",
                    "low_section_speed": "Low Section Speed",
                    "high_section_speed": "High Section Speed",
                    "molten_volume": "Molten Volume",
                    "cast_pressure": "Cast Pressure",
                    "biscuit_thickness": "Biscuit Thickness",
                    "upper_mold_temp1": "Upper Mold Temp 1",
                    "upper_mold_temp2": "Upper Mold Temp 2",
                    "upper_mold_temp3": "Upper Mold Temp 3",
                    "lower_mold_temp1": "Lower Mold Temp 1",
                    "lower_mold_temp2": "Lower Mold Temp 2",
                    "lower_mold_temp3": "Lower Mold Temp 3",
                    "sleeve_temperature": "Sleeve Temperature",
                    "physical_strength": "Physical Strength",
                    "Coolant_temperature": "Coolant Temperature"
                },
                selected="physical_strength",
                inline=False
            ),
        ),
        open=False
    ),
    
    # Xbar 관리도
    ui.card(
        ui.card_header("Xbar Control Chart (Mean)"),
        ui.output_plot("plot_xbar_chart", height="600px"),
    ),
    
    # R 관리도
    ui.card(
        ui.card_header("R Control Chart (Range)"),
        ui.output_plot("plot_r_chart", height="600px"),
    )
)

# 탭별 서버
def tab_server(input, output, session):
    
    @render.plot
    def plot_p_chart():
        # 필터 적용 (버튼 클릭 시)
        input.apply_filter()
        
        # 분석 모드 확인
        analysis_mode = input.analysis_mode()
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = input.date_range()
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start, date_end)
            
            # 날짜 기반 관리도 계산
            date_stats, p_bar, UCL, LCL = calculate_p_chart_by_date(df)
            display_data = date_stats
            x_column = 'date_only'
            x_label = 'Date'
            title = f'P Control Chart (Date-based)'
        else:
            # 서브그룹 기반 분석 (n=5 고정)
            df = load_and_filter_data()
            subgroup_size = 5
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, p_bar, UCL, LCL = calculate_p_chart(df, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, input.subgroup_start())
            end = min(len(subgroup_stats), input.subgroup_end())
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = 'Subgroup Number'
            title = f'P Control Chart (Subgroup-based, n=5)'
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 자동 스케일링 완전 비활성화
        ax.autoscale(enable=False)
        
        # Y축 범위 0~1로 강제 고정
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # X축 범위 설정
        if len(display_data) > 0 and analysis_mode == "subgroup":
            ax.set_xlim(display_data[x_column].min() - 1, display_data[x_column].max() + 1)
        
        # 2시그마 경고선 계산
        warn_upper = p_bar + (UCL - p_bar) * 2/3
        warn_lower = p_bar - (p_bar - LCL) * 2/3
        
        # 관리한계 상태 판정
        out_of_control = (display_data['p'] > UCL) | (display_data['p'] < LCL)
        
        # 전체 데이터를 먼저 선으로 연결
        ax.plot(display_data[x_column], display_data['p'], 
                color='#2E86AB', linewidth=2, linestyle='-', zorder=2)
        
        # 정상 포인트 (파란색)
        ax.scatter(display_data.loc[~out_of_control, x_column], 
                  display_data.loc[~out_of_control, 'p'],
                  color='#2E86AB', s=50, marker='o', 
                  label='Defect Rate (P)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control.any():
            ax.scatter(display_data.loc[out_of_control, x_column], 
                      display_data.loc[out_of_control, 'p'],
                      color='red', s=80, marker='o', 
                      label='Out of Control', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=p_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'CL (P̄ = {p_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL, color='red', linestyle='--', linewidth=2, 
                   label=f'UCL = {UCL:.4f}', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL, color='red', linestyle='--', linewidth=2, 
                   label=f'LCL = {LCL:.4f}', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (+2σ = {warn_upper:.4f})', zorder=1)
        ax.axhline(y=warn_lower, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (-2σ = {warn_lower:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Defect Rate (P)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)
        
        # Y축 여백 완전 제거
        ax.margins(y=0)
        
        plt.tight_layout()
        
        return fig
    
    @render.plot
    def plot_xbar_chart():
        # 필터 적용
        input.apply_filter()
        
        # 분석 모드 확인
        analysis_mode = input.analysis_mode()
        variable = input.xbar_variable()
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = input.date_range()
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start, date_end)
            
            # 날짜 기반 관리도 계산
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = 'Date'
            title = f'Xbar Control Chart (Date-based)'
        else:
            # 서브그룹 기반 분석 (n=5 고정)
            df = load_and_filter_data()
            subgroup_size = 5
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, input.subgroup_start())
            end = min(len(subgroup_stats), input.subgroup_end())
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = 'Subgroup Number'
            title = f'Xbar Control Chart (Subgroup-based, n=5)'
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 2시그마 경고선 계산
        warn_upper_xbar = xbar_bar + (UCL_xbar - xbar_bar) * 2/3
        warn_lower_xbar = xbar_bar - (xbar_bar - LCL_xbar) * 2/3
        
        # 관리한계 상태 판정
        out_of_control_xbar = (display_data['mean'] > UCL_xbar) | (display_data['mean'] < LCL_xbar)
        
        # 전체 데이터를 먼저 선으로 연결
        ax.plot(display_data[x_column], display_data['mean'], 
                color='#2E86AB', linewidth=2, linestyle='-', zorder=2)
        
        # 정상 포인트 (파란색)
        ax.scatter(display_data.loc[~out_of_control_xbar, x_column], 
                  display_data.loc[~out_of_control_xbar, 'mean'],
                  color='#2E86AB', s=50, marker='o', 
                  label='Xbar (Mean)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control_xbar.any():
            ax.scatter(display_data.loc[out_of_control_xbar, x_column], 
                      display_data.loc[out_of_control_xbar, 'mean'],
                      color='red', s=80, marker='o', 
                      label='Out of Control', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=xbar_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'CL (X̿ = {xbar_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL_xbar, color='red', linestyle='--', linewidth=2, 
                   label=f'UCL = {UCL_xbar:.4f}', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL_xbar, color='red', linestyle='--', linewidth=2, 
                   label=f'LCL = {LCL_xbar:.4f}', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper_xbar, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (+2σ = {warn_upper_xbar:.4f})', zorder=1)
        ax.axhline(y=warn_lower_xbar, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (-2σ = {warn_lower_xbar:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable} - Mean', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    @render.plot
    def plot_r_chart():
        # 필터 적용
        input.apply_filter()
        
        # 분석 모드 확인
        analysis_mode = input.analysis_mode()
        variable = input.xbar_variable()
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = input.date_range()
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start, date_end)
            
            # 날짜 기반 관리도 계산
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = 'Date'
            title = f'R Control Chart (Date-based)'
        else:
            # 서브그룹 기반 분석 (n=5 고정)
            df = load_and_filter_data()
            subgroup_size = 5
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, input.subgroup_start())
            end = min(len(subgroup_stats), input.subgroup_end())
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = 'Subgroup Number'
            title = f'R Control Chart (Subgroup-based, n=5)'
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 2시그마 경고선 계산
        warn_upper_r = r_bar + (UCL_r - r_bar) * 2/3
        warn_lower_r = r_bar - (r_bar - LCL_r) * 2/3
        
        # 관리한계 상태 판정
        out_of_control_r = (display_data['range'] > UCL_r) | (display_data['range'] < LCL_r)
        
        # 전체 데이터를 먼저 선으로 연결
        ax.plot(display_data[x_column], display_data['range'], 
                color='#E27D60', linewidth=2, linestyle='-', zorder=2)
        
        # 정상 포인트 (주황색)
        ax.scatter(display_data.loc[~out_of_control_r, x_column], 
                  display_data.loc[~out_of_control_r, 'range'],
                  color='#E27D60', s=50, marker='o', 
                  label='R (Range)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control_r.any():
            ax.scatter(display_data.loc[out_of_control_r, x_column], 
                      display_data.loc[out_of_control_r, 'range'],
                      color='red', s=80, marker='o', 
                      label='Out of Control', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=r_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'CL (R̄ = {r_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL_r, color='red', linestyle='--', linewidth=2, 
                   label=f'UCL = {UCL_r:.4f}', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL_r, color='red', linestyle='--', linewidth=2, 
                   label=f'LCL = {LCL_r:.4f}', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper_r, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (+2σ = {warn_upper_r:.4f})', zorder=1)
        ax.axhline(y=warn_lower_r, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'Warning (-2σ = {warn_lower_r:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable} - Range', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig