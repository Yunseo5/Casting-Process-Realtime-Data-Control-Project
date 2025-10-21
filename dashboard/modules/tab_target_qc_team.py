# modules/tab_target_qc_team.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import joblib  # type: ignore
except ImportError:
    joblib = None

try:
    import shap  # type: ignore
except ImportError:
    shap = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_ARTIFACT_PATH = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"
PI_TABLE_PATH = BASE_DIR / "reports" / "permutation_importance_valid.csv"

PDP_GRID_SIZE = 15
ICE_SAMPLE_SIZE = 25
MODEL_SAMPLE_SIZE = 600
SHAP_TOP_K = 10
MCI_THRESHOLD = 0.2
MCI_QUANTILES = (0.05, 0.95)
CPK_STATUS_RULES = [
    ("안정", 1.33),
    ("주의", 1.0),
]


def read_raw_data():
    df = pd.read_csv(DATA_PATH)
    df['registration_time'] = pd.to_datetime(df['registration_time'])
    return df


try:
    _mold_df = read_raw_data()
    MOLD_CODE_CHOICES = sorted(_mold_df['mold_code'].dropna().astype(str).unique().tolist())
    del _mold_df
except (FileNotFoundError, KeyError):
    MOLD_CODE_CHOICES = []

# UI에 표시할 변수 한글명 매핑
VARIABLE_LABELS = {
    "molten_temp": "용탕 온도",
    "low_section_speed": "하부 구간 속도",
    "high_section_speed": "상부 구간 속도",
    "molten_volume": "용탕량",
    "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스킷 두께",
    "upper_mold_temp1": "상부 금형 온도 1",
    "upper_mold_temp2": "상부 금형 온도 2",
    "upper_mold_temp3": "상부 금형 온도 3",
    "lower_mold_temp1": "하부 금형 온도 1",
    "lower_mold_temp2": "하부 금형 온도 2",
    "lower_mold_temp3": "하부 금형 온도 3",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "EMS 작동 시간",
}

# 데이터 로드 및 필터링 함수
def load_and_filter_data(date_start=None, date_end=None, mold_codes=None):
    df = read_raw_data()

    # 금형 코드 필터링
    if mold_codes:
        if 'mold_code' not in df.columns:
            raise ValueError("mold_code 컬럼이 없습니다.")
        df = df[df['mold_code'].astype(str).isin(mold_codes)]
    
    # 날짜 필터링 (registration_time 기준)
    if date_start is not None:
        df = df[df['registration_time'].dt.date >= pd.to_datetime(date_start).date()]
    if date_end is not None:
        df = df[df['registration_time'].dt.date <= pd.to_datetime(date_end).date()]

    return df


def load_model_artifact():
    if joblib is None or not MODEL_ARTIFACT_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_ARTIFACT_PATH)
    except Exception:
        return None


def derive_model_metadata():
    metadata = {
        "numeric": [],
        "categorical": [],
        "all": [],
        "numeric_defaults": {},
        "categorical_defaults": {},
    }

    try:
        df = read_raw_data()
    except Exception:
        return metadata

    feature_df = df.drop(columns=["passorfail", "registration_time"], errors="ignore").copy()
    metadata["all"] = feature_df.columns.tolist()

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    metadata["numeric"] = numeric_cols
    metadata["categorical"] = [col for col in metadata["all"] if col not in numeric_cols]

    if numeric_cols:
        numeric_defaults = (
            feature_df[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .median()
            .fillna(0.0)
        )
        metadata["numeric_defaults"] = numeric_defaults.to_dict()

    cat_defaults = {}
    for col in metadata["categorical"]:
        series = feature_df[col].dropna()
        if not series.empty:
            cat_defaults[col] = series.mode().iloc[0]
        else:
            cat_defaults[col] = "<NA>"
    metadata["categorical_defaults"] = cat_defaults

    return metadata


def synchronize_model_metadata(metadata, bundle):
    if bundle is None:
        return metadata

    numeric_cols = metadata.get("numeric", [])
    categorical_cols = metadata.get("categorical", [])

    scaler = bundle.get("scaler")
    ordinal_encoder = bundle.get("ordinal_encoder")

    if scaler is not None:
        scaler_cols = getattr(scaler, "feature_names_in_", None)
        if scaler_cols is not None:
            numeric_cols = list(scaler_cols)
    if ordinal_encoder is not None:
        encoder_cols = getattr(ordinal_encoder, "feature_names_in_", None)
        if encoder_cols is not None:
            categorical_cols = list(encoder_cols)

    metadata["numeric"] = list(numeric_cols)
    metadata["categorical"] = list(categorical_cols)

    numeric_defaults = dict(metadata.get("numeric_defaults", {}))
    metadata["numeric_defaults"] = {
        col: numeric_defaults.get(col, 0.0) for col in metadata["numeric"]
    }
    categorical_defaults = dict(metadata.get("categorical_defaults", {}))
    metadata["categorical_defaults"] = {
        col: categorical_defaults.get(col, "<NA>") for col in metadata["categorical"]
    }

    metadata["all"] = list(dict.fromkeys(metadata["numeric"] + metadata["categorical"]))
    return metadata


def transform_for_model(df, metadata, bundle):
    if bundle is None:
        return None, None, "모델 아티팩트를 불러오지 못했습니다."

    numeric_cols = metadata.get("numeric", [])
    categorical_cols = metadata.get("categorical", [])
    all_columns = metadata.get("all", [])

    if not all_columns:
        return None, None, "모델 입력 컬럼 정보가 비어 있습니다."

    df_features = df.copy()
    for col in all_columns:
        if col not in df_features.columns:
            if col in numeric_cols:
                df_features[col] = metadata["numeric_defaults"].get(col, 0.0)
            else:
                df_features[col] = metadata["categorical_defaults"].get(col, "<NA>")

    df_features = df_features[all_columns]

    scaler = bundle.get("scaler")
    ordinal_encoder = bundle.get("ordinal_encoder")
    onehot_encoder = bundle.get("onehot_encoder")

    if numeric_cols:
        num_frame = df_features[numeric_cols].apply(pd.to_numeric, errors="coerce")
        fill_map = {col: metadata["numeric_defaults"].get(col, 0.0) for col in numeric_cols}
        num_frame = num_frame.fillna(fill_map)
        if scaler is not None:
            try:
                num_array = scaler.transform(num_frame.to_numpy())
            except Exception as exc:
                return None, None, f"수치형 스케일링 실패: {exc}"
        else:
            num_array = num_frame.to_numpy(dtype=float)
    else:
        num_array = np.empty((len(df_features), 0))

    ohe_feature_names = []
    if categorical_cols:
        if ordinal_encoder is None or onehot_encoder is None:
            return None, None, "범주형 인코더가 누락되었습니다."
        cat_frame = (
            df_features[categorical_cols]
            .fillna({col: metadata["categorical_defaults"].get(col, "<NA>") for col in categorical_cols})
            .astype(str)
        )
        try:
            cat_ord = ordinal_encoder.transform(cat_frame)
            cat_ohe = onehot_encoder.transform(cat_ord)
        except Exception as exc:
            return None, None, f"범주형 인코딩 실패: {exc}"
        try:
            ohe_feature_names = onehot_encoder.get_feature_names_out(categorical_cols).tolist()
        except AttributeError:
            ohe_feature_names = [f"{col}_{idx}" for col in categorical_cols for idx in range(cat_ohe.shape[1])]
    else:
        cat_ohe = np.empty((len(df_features), 0))

    if num_array.size or cat_ohe.size:
        feature_matrix = np.hstack([arr for arr in (num_array, cat_ohe) if arr.size])
    else:
        feature_matrix = np.zeros((len(df_features), 0))

    feature_names = list(numeric_cols) + list(ohe_feature_names)
    return feature_matrix, feature_names, None


def predict_proba_with_model(df, metadata, bundle):
    model = bundle.get("model") if bundle else None
    if model is None:
        return None, None, "LightGBM 모델을 찾을 수 없습니다."

    feature_matrix, feature_names, error = transform_for_model(df, metadata, bundle)
    if error:
        return None, None, error

    try:
        probs = model.predict(feature_matrix)
    except Exception as exc:
        return None, None, f"예측 실패: {exc}"

    return probs, feature_names, None


def compute_pdp_curve(df, variable, metadata, bundle, grid_size=PDP_GRID_SIZE, sample_size=MODEL_SAMPLE_SIZE):
    if variable not in metadata.get("all", []):
        return None, f"'{variable}' 컬럼을 찾을 수 없습니다."

    if df.empty:
        return None, "분석할 데이터가 없습니다."

    sample_df = df[metadata.get("all", [])].copy()
    sample_df = sample_df.sample(n=min(sample_size, len(sample_df)), random_state=41)

    if variable not in metadata.get("numeric", []):
        return None, "현재 PDP 계산은 수치형 변수만 지원합니다."

    series = pd.to_numeric(sample_df[variable], errors="coerce").dropna()
    if series.empty:
        return None, "선택한 변수에 유효한 값이 없습니다."

    low = series.quantile(0.05)
    high = series.quantile(0.95)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = series.min(), series.max()

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None, "선택한 변수의 범위가 충분하지 않습니다."

    grid = np.linspace(low, high, grid_size)
    averages = []
    for value in grid:
        temp = sample_df.copy()
        temp[variable] = value
        probs, _, error = predict_proba_with_model(temp, metadata, bundle)
        if error:
            return None, error
        averages.append(float(np.mean(probs)))

    return {"grid": grid.tolist(), "averages": averages}, None


def find_largest_safe_interval(grid, averages, threshold):
    if not grid or not averages:
        return None

    pairs = sorted(
        [
            (float(x), float(y))
            for x, y in zip(grid, averages)
            if np.isfinite(x) and np.isfinite(y)
        ],
        key=lambda item: item[0],
    )

    segments = []
    start = None
    last_val = None
    for x, y in pairs:
        if y <= threshold:
            if start is None:
                start = x
            last_val = x
        else:
            if start is not None and last_val is not None:
                segments.append((start, last_val))
                start = None
                last_val = None

    if start is not None and last_val is not None:
        segments.append((start, last_val))

    if not segments:
        return None

    segments.sort(key=lambda seg: seg[1] - seg[0], reverse=True)
    safe_min, safe_max = segments[0]
    if safe_min == safe_max:
        return None
    return safe_min, safe_max


MODEL_METADATA = synchronize_model_metadata(derive_model_metadata(), load_model_artifact())
_MODEL_BUNDLE_CACHE = {"bundle": None, "loaded": False}


def get_model_bundle():
    if not _MODEL_BUNDLE_CACHE["loaded"]:
        _MODEL_BUNDLE_CACHE["bundle"] = load_model_artifact()
        _MODEL_BUNDLE_CACHE["loaded"] = True
    bundle = _MODEL_BUNDLE_CACHE["bundle"]
    if bundle is not None:
        synchronize_model_metadata(MODEL_METADATA, bundle)
    return bundle


MCI_CONFIGS = [
    {
        "id": "mci_melt",
        "variable": "molten_temp",
        "label": "녹이기 (용탕 온도)",
        "threshold": MCI_THRESHOLD,
    },
    {
        "id": "mci_pour",
        "variable": "cast_pressure",
        "label": "주입 (주조 압력)",
        "threshold": MCI_THRESHOLD,
    },
    {
        "id": "mci_cool",
        "variable": "upper_mold_temp1",
        "label": "냉각 (상부 금형 온도1)",
        "threshold": MCI_THRESHOLD,
    },
]

def evaluate_mci_metric(df, variable, label, bundle, metadata, threshold=MCI_THRESHOLD, quantiles=MCI_QUANTILES):
    result = {
        "variable": variable,
        "label": label,
        "cpk": None,
        "status": "데이터 부족",
        "safe_min": None,
        "safe_max": None,
        "actual_min": None,
        "actual_max": None,
        "mean": None,
        "std": None,
        "details": None,
    }

    if df.empty:
        result["details"] = "분석할 데이터가 없습니다."
        return result

    if variable not in df.columns:
        result["details"] = f"'{variable}' 컬럼을 찾을 수 없습니다."
        return result

    if variable not in metadata.get("numeric", []):
        result["details"] = "현재 MCI 계산은 수치형 변수만 지원합니다."
        return result

    pdp_payload, error = compute_pdp_curve(df, variable, metadata, bundle)
    if error:
        result["details"] = error
        return result

    safe_range = find_largest_safe_interval(
        pdp_payload["grid"],
        pdp_payload["averages"],
        threshold,
    )

    series = pd.to_numeric(df[variable], errors="coerce").dropna()
    if series.empty:
        result["details"] = "실제 공정 데이터가 부족합니다."
        return result

    actual_min, actual_max = series.quantile(list(quantiles))
    if not np.isfinite(actual_min) or not np.isfinite(actual_max):
        result["details"] = "실제 공정 범위를 계산할 수 없습니다."
        return result

    actual_min = float(actual_min)
    actual_max = float(actual_max)
    result["actual_min"] = actual_min
    result["actual_max"] = actual_max

    if safe_range is None:
        result["status"] = "위험"
        result["details"] = "허용 구간을 찾지 못했습니다."
        return result
    safe_min, safe_max = safe_range
    result["safe_min"] = safe_min
    result["safe_max"] = safe_max

    mean = float(series.mean())
    std = float(series.std(ddof=1))
    result["mean"] = mean
    result["std"] = std

    if not np.isfinite(std) or std <= 0:
        result["status"] = "위험"
        result["details"] = "표준편차를 계산할 수 없습니다."
        return result

    cpu = (safe_max - mean) / (3 * std)
    cpl = (mean - safe_min) / (3 * std)
    cpk = min(cpu, cpl)
    result["cpk"] = cpk

    if cpk < 0:
        result["status"] = "위험"
        result["details"] = "평균이 허용 구간 밖에 있습니다."
        return result

    for status, threshold_value in CPK_STATUS_RULES:
        if cpk >= threshold_value:
            result["status"] = status
            break
    else:
        result["status"] = "위험"
    result["details"] = None

    return result


def summarize_mci_status(cpk_value):
    if cpk_value is None:
        return "데이터 부족"
    if cpk_value < 0:
        return "위험"
    for status, threshold_value in CPK_STATUS_RULES:
        if cpk_value >= threshold_value:
            return status
    return "위험"


def compute_mci_metrics(df, metadata, bundle):
    if bundle is None:
        return {"error": "모델 아티팩트를 찾을 수 없습니다.", "metrics": [], "overall_ratio": None}

    metrics = []
    for config in MCI_CONFIGS:
        metric = evaluate_mci_metric(
            df,
            config["variable"],
            config["label"],
            bundle,
            metadata,
            threshold=config.get("threshold", MCI_THRESHOLD),
        )
        metrics.append(metric)

    valid_cpk = [m["cpk"] for m in metrics if m["cpk"] is not None]
    if valid_cpk:
        overall_cpk = min(valid_cpk)
        overall_status = summarize_mci_status(overall_cpk)
    else:
        overall_cpk = None
        overall_status = "데이터 부족"

    return {
        "metrics": metrics,
        "overall_cpk": overall_cpk,
        "overall_status": overall_status,
        "error": None,
    }

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
    ui.h2("품질 관리팀 - 관리도", class_="mb-4"),
    ui.tags.style("""
    .mci-metric-card{padding:10px 12px;border-radius:12px;min-height:110px}
    .mci-metric-card h3{font-size:1.35rem;margin-bottom:2px;font-weight:700}
    .mci-value-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
    .mci-status-badge{font-weight:800;font-size:3rem;margin-left:8px;line-height:1}
    .mci-metric-card .mci-detail{font-size:0.75rem;color:#6c757d;margin:0}
    .mci-overall-card{background:#f7f9fc}
    .mci-cards-row{gap:12px}
    """),

    ui.output_ui("mci_overview"),
    
    ui.card(
        ui.card_header("필터 설정"),
        ui.tags.style("""
        #filter-section .form-label {
            white-space: normal;
        }
        #filter-section .layout-columns {
            gap: 16px !important;
        }
        #filter-section .shiny-input-container {
            width: 100%;
        }
        #filter-section .selectize-input {
            padding: 10px !important;
            min-height: 46px !important;
            border-radius: 8px !important;
        }
        """),
        ui.div(
            ui.layout_columns(
                ui.input_radio_buttons(
                    "analysis_mode",
                    "분석 모드",
                    choices={
                        "subgroup": "서브그룹 기반 분석",
                        "date": "날짜 기반 분석"
                    },
                    selected="subgroup",
                    inline=True
                ),
                ui.input_selectize(
                    "mold_code",
                    "금형 코드",
                    choices=MOLD_CODE_CHOICES,
                    multiple=True,
                    options={
                        "placeholder": "금형 코드를 선택하세요",
                        "plugins": ["remove_button"]
                    }
                ),
                ui.input_selectize(
                    "xbar_variable",
                    "Xbar-R 변수",
                    choices=VARIABLE_LABELS,
                    selected="physical_strength",
                    options={
                        "placeholder": "관리도에 사용할 변수를 선택하세요",
                        "dropdownParent": "body"
                    },
                    width="100%"
                ),
                col_widths=[4, 4, 4]
            ),
            class_="mb-3",
            id="filter-section"
        ),
        ui.panel_conditional(
            "input.analysis_mode === 'subgroup'",
            ui.layout_columns(
                ui.input_slider(
                    "subgroup_size",
                    "서브그룹 크기 (n = 1~10)",
                    min=1,
                    max=10,
                    value=5,
                    step=1,
                    width="100%"
                ),
                ui.input_numeric("subgroup_start", "시작 서브그룹", value=0, min=0, width="100%"),
                ui.input_numeric("subgroup_end", "종료 서브그룹", value=100, min=1, width="100%"),
                col_widths=[6, 3, 3],
                class_="align-items-end"
            )
        ),
        ui.panel_conditional(
            "input.analysis_mode === 'date'",
            ui.input_date_range(
                "date_range",
                "날짜 범위 (2019-01-02 ~ 2019-03-12)",
                start="2019-01-02",
                end="2019-03-12",
                min="2019-01-02",
                max="2019-03-12",
                width="100%"
            ),
        ),
        ui.tags.p(
            "선택한 변수 기준으로 P, Xbar, R 관리도가 계산됩니다.",
            class_="text-muted small mb-2"
        ),
        ui.div(
            ui.input_action_button(
                "apply_filter",
                "필터 적용",
                class_="btn-primary"
            ),
            class_="d-flex justify-content-end"
        ),
    ),

    ui.card(
        ui.card_header("P 관리도 (불량률)"),
        ui.output_plot("plot_p_chart", height="550px"),
    ),
    
    ui.layout_columns(
        ui.card(
            ui.card_header("Xbar 관리도 (평균)"),
            ui.output_plot("plot_xbar_chart", height="620px"),
            style="min-height: 680px;"
        ),
        ui.card(
            ui.card_header("R 관리도 (범위)"),
            ui.output_plot("plot_r_chart", height="620px"),
            style="min-height: 680px;"
        ),
        col_widths=[6, 6]
    )
)

# 탭별 서버
def tab_server(input, output, session):
    
    def current_filters():
        with reactive.isolate():
            analysis_mode = input.analysis_mode()
            mold_selection = input.mold_code()
            mold_codes = list(mold_selection) if mold_selection else []
            date_range = input.date_range()
            subgroup_size_value = input.subgroup_size()
            subgroup_start_value = input.subgroup_start()
            subgroup_end_value = input.subgroup_end()
            variable = input.xbar_variable()

        subgroup_size = int(subgroup_size_value) if subgroup_size_value is not None else 5
        subgroup_start = int(subgroup_start_value) if subgroup_start_value is not None else 0
        subgroup_end = int(subgroup_end_value) if subgroup_end_value is not None else subgroup_start + 100

        return {
            "analysis_mode": analysis_mode,
            "mold_codes": mold_codes,
            "date_range": date_range,
            "subgroup_size": subgroup_size,
            "subgroup_start": subgroup_start,
            "subgroup_end": subgroup_end,
            "variable": variable,
        }

    @reactive.Calc
    def mci_dataset():
        filters = current_filters()
        mold_codes = filters["mold_codes"] or None
        date_range = filters["date_range"]
        date_start = date_range[0] if date_range else None
        date_end = date_range[1] if date_range else None

        df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
        if df.empty:
            return pd.DataFrame()
        return df.reset_index(drop=True)

    @reactive.Calc
    def mci_metrics():
        input.apply_filter()
        bundle = get_model_bundle()
        df = mci_dataset()
        return compute_mci_metrics(df, MODEL_METADATA, bundle)

    @render.ui
    def mci_overview():
        metrics_payload = mci_metrics()
        error = metrics_payload.get("error")
        if error:
            return ui.card(ui.card_body(ui.p(error, class_="text-danger")), class_="mb-3")

        status_colors = {
            "안정": "#2ecc71",
            "주의": "#f39c12",
            "위험": "#e74c3c",
            "데이터 부족": "#95a5a6",
        }

        def format_cpk(value):
            if value is None:
                return "Cpk --"
            return f"Cpk {value:.2f}"

        def format_range(min_val, max_val):
            if min_val is None or max_val is None:
                return "데이터 없음"
            return f"[{min_val:.2f}, {max_val:.2f}]"

        def format_sigma(value):
            if value is None or not np.isfinite(value):
                return "σ: 계산 불가"
            return f"σ: {value:.3f}"

        cards = []

        overall_cpk = metrics_payload.get("overall_cpk")
        overall_status = metrics_payload.get("overall_status", "데이터 부족")
        overall_color = status_colors.get(overall_status, "#95a5a6")

        cards.append(
            ui.card(
                ui.card_body(
                    ui.h6("모델 기반 Cpk (최소)", class_="text-muted text-uppercase mb-2"),
                    ui.div(
                        ui.h3(format_cpk(overall_cpk), class_="mb-0"),
                        ui.span(overall_status, class_="mci-status-badge", style=f"color:{overall_color};"),
                        class_="mci-value-row",
                    ),
                ),
                class_="mci-metric-card mci-overall-card",
            )
        )

        for metric in metrics_payload.get("metrics", []):
            status = metric.get("status", "데이터 부족")
            color = status_colors.get(status, "#95a5a6")

            cards.append(
                ui.card(
                    ui.card_body(
                        ui.h6(metric.get("label", metric.get("variable", "")), class_="text-muted text-uppercase mb-2"),
                        ui.div(
                            ui.h3(format_cpk(metric.get("cpk")), class_="mb-0"),
                            ui.span(status, class_="mci-status-badge", style=f"color:{color};"),
                            class_="mci-value-row",
                        ),
                        ui.p(
                            f"허용 {format_range(metric.get('safe_min'), metric.get('safe_max'))} · 실제 {format_range(metric.get('actual_min'), metric.get('actual_max'))}",
                            class_="mci-detail",
                        ),
                        ui.p(
                            (
                                f"μ {metric.get('mean'):.2f}"
                                if metric.get("mean") is not None and np.isfinite(metric.get("mean"))
                                else "μ 계산 불가"
                            )
                            + " · "
                            + format_sigma(metric.get("std")),
                            class_="mci-detail",
                        ),
                        ui.p(metric.get("details", ""), class_="mci-detail") if metric.get("details") else ui.div(),
                    ),
                    class_="mci-metric-card",
                )
            )

        return ui.div(
            ui.layout_columns(*cards, col_widths=[3] * len(cards)),
            class_="mb-3 mci-cards-row",
        )

    @render.plot
    def plot_p_chart():
        # 필터 적용 (버튼 클릭 시)
        input.apply_filter()
        
        filters = current_filters()
        
        # 금형 코드 선택값
        mold_codes = filters["mold_codes"] or None
        
        # 분석 모드 확인
        analysis_mode = filters["analysis_mode"]
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            
            # 날짜 기반 관리도 계산
            date_stats, p_bar, UCL, LCL = calculate_p_chart_by_date(df)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = f'P 관리도 (날짜 기반)'
        else:
            # 서브그룹 기반 분석 (슬라이더로 조정 가능)
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, p_bar, UCL, LCL = calculate_p_chart(df, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'P 관리도 (서브그룹 기반, n={subgroup_size})'
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
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
                  color='#2E86AB', s=45, marker='o',
                  label='불량률 (P)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control.any():
            ax.scatter(display_data.loc[out_of_control, x_column], 
                      display_data.loc[out_of_control, 'p'],
                      color='red', s=55, marker='o',
                      label='관리한계 초과', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=p_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'중심선 (P̄ = {p_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL, color='red', linestyle='--', linewidth=2, 
                   label=f'상한 관리한계선 (UCL = {UCL:.4f})', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL, color='red', linestyle='--', linewidth=2, 
                   label=f'하한 관리한계선 (LCL = {LCL:.4f})', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (+2σ = {warn_upper:.4f})', zorder=1)
        ax.axhline(y=warn_lower, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (-2σ = {warn_lower:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('불량률 (P)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85, borderpad=0.6, labelspacing=0.4)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        return fig
    
    @render.plot
    def plot_xbar_chart():
        # 필터 적용
        input.apply_filter()
        
        filters = current_filters()
        
        # 분석 모드 확인
        analysis_mode = filters["analysis_mode"]
        variable = filters["variable"]
        variable_label = VARIABLE_LABELS.get(variable, variable)
        mold_codes = filters["mold_codes"] or None
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            
            # 날짜 기반 관리도 계산
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = f'Xbar 관리도 (날짜 기반)'
        else:
            # 서브그룹 기반 분석 (슬라이더로 조정 가능)
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'Xbar 관리도 (서브그룹 기반, n={subgroup_size})'
        
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
                  color='#2E86AB', s=45, marker='o',
                  label='Xbar (평균)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control_xbar.any():
            ax.scatter(display_data.loc[out_of_control_xbar, x_column], 
                      display_data.loc[out_of_control_xbar, 'mean'],
                      color='red', s=55, marker='o',
                      label='관리한계 초과', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=xbar_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'중심선 (X̿ = {xbar_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL_xbar, color='red', linestyle='--', linewidth=2, 
                   label=f'상한 관리한계선 (UCL = {UCL_xbar:.4f})', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL_xbar, color='red', linestyle='--', linewidth=2, 
                   label=f'하한 관리한계선 (LCL = {LCL_xbar:.4f})', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper_xbar, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (+2σ = {warn_upper_xbar:.4f})', zorder=1)
        ax.axhline(y=warn_lower_xbar, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (-2σ = {warn_lower_xbar:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable_label} - 평균', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            fontsize=8,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5
        )
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.28)

        return fig
    
    @render.plot
    def plot_r_chart():
        # 필터 적용
        input.apply_filter()
        
        filters = current_filters()

        # 분석 모드 확인
        analysis_mode = filters["analysis_mode"]
        variable = filters["variable"]
        variable_label = VARIABLE_LABELS.get(variable, variable)
        mold_codes = filters["mold_codes"] or None
        
        if analysis_mode == "date":
            # 날짜 기반 분석
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            
            # 데이터 로드 및 필터링
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            
            # 날짜 기반 관리도 계산
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = f'R 관리도 (날짜 기반)'
        else:
            # 서브그룹 기반 분석 (슬라이더로 조정 가능)
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            
            # 서브그룹 기반 관리도 계산
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = \
                calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            
            # 구간 선택 적용
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'R 관리도 (서브그룹 기반, n={subgroup_size})'
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 2시그마 경고선 계산
        warn_upper_r = r_bar + (UCL_r - r_bar) * 2/3
        warn_lower_r = r_bar - (r_bar - LCL_r) * 2/3
        
        # 관리한계 상태 판정
        out_of_control_r = (display_data['range'] > UCL_r) | (display_data['range'] < LCL_r)
        
        # 전체 데이터를 먼저 선으로 연결
        ax.plot(display_data[x_column], display_data['range'],
                color="#2E86AB", linewidth=2, linestyle='-', zorder=2)
        
        # 정상 포인트 (주황색)
        ax.scatter(display_data.loc[~out_of_control_r, x_column],
                  display_data.loc[~out_of_control_r, 'range'],
                  color='#2E86AB', s=45, marker='o',
                  label='R (범위)', zorder=3)
        
        # 이탈 포인트 (빨간색, 크게)
        if out_of_control_r.any():
            ax.scatter(display_data.loc[out_of_control_r, x_column],
                      display_data.loc[out_of_control_r, 'range'],
                      color='red', s=55, marker='o',
                      label='관리한계 초과', zorder=4)
        
        # 중심선 (CL)
        ax.axhline(y=r_bar, color='green', linestyle='-', linewidth=2, 
                   label=f'중심선 (R̄ = {r_bar:.4f})', zorder=1)
        
        # 상한 관리한계선 (UCL)
        ax.axhline(y=UCL_r, color='red', linestyle='--', linewidth=2, 
                   label=f'상한 관리한계선 (UCL = {UCL_r:.4f})', zorder=1)
        
        # 하한 관리한계선 (LCL)
        ax.axhline(y=LCL_r, color='red', linestyle='--', linewidth=2, 
                   label=f'하한 관리한계선 (LCL = {LCL_r:.4f})', zorder=1)
        
        # 2시그마 경고선
        ax.axhline(y=warn_upper_r, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (+2σ = {warn_upper_r:.4f})', zorder=1)
        ax.axhline(y=warn_lower_r, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'경고선 (-2σ = {warn_lower_r:.4f})', zorder=1)
        
        # 그래프 설정
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable_label} - 범위', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            fontsize=8,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5
        )
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        # 날짜 모드일 경우 x축 회전
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.28)

        return fig
