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
CONTROL_LIMITS_PATH = BASE_DIR / "data" / "관리도평균_표준편차.csv"

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
# Cpk 표시 옵션: True면 카드/개요에서 |Cpk|로 표시 (계산값은 그대로 유지)
CPK_DISPLAY_FLOOR = 0.03
DISPLAY_ABS_CPK = False 

# 👉 변경/추가: (선택) 엔지니어링 규격 사양. 있으면 채워서 사용, 없으면 빈 dict 유지하면
#             아래에서 PDP 기반 safe range를 '가상 규격'으로 사용함(해석 라벨링 필요).
LSL_USL_MAP = {
    # "sleeve_temperature": {"LSL": 45.0, "USL": 55.0},
    # "molten_temp": {"LSL": 660.0, "USL": 690.0},
}

def read_raw_data():
    df = pd.read_csv(DATA_PATH)
    df['registration_time'] = pd.to_datetime(df['registration_time'])
    return df


def load_control_limits():
    """관리한계 평균 및 표준편차 CSV 파일 로드"""
    try:
        df = pd.read_csv(CONTROL_LIMITS_PATH)
        return df
    except FileNotFoundError:
        return None


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

    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
    ordinal_encoder = bundle.get("ordinal_encoder") if isinstance(bundle, dict) else None

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

    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
    ordinal_encoder = bundle.get("ordinal_encoder") if isinstance(bundle, dict) else None
    onehot_encoder = bundle.get("onehot_encoder") if isinstance(bundle, dict) else None

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
    model = bundle.get("model") if isinstance(bundle, dict) else None
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

# 👉 변경/추가: 군내 σ(R̄/d2) 추정 헬퍼
def _estimate_within_sigma_via_rbar(series, n=5):
    """서브그룹 크기 n으로 R̄/d2 기반 군내 σ 추정. 실패 시 None."""
    try:
        x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        n = max(2, int(n))
        k = len(x) // n
        if k < 1:
            return None
        x = x[: k * n].reshape(k, n)
        rbar = x.ptp(axis=1).mean()
        d2_table = {2:1.128,3:1.693,4:2.059,5:2.326,6:2.534,7:2.704,8:2.847,9:2.970,10:3.078}
        d2 = d2_table.get(n, 2.326)  # 기본값 n=5
        if not np.isfinite(rbar) or d2 <= 0:
            return None
        return float(rbar / d2)
    except Exception:
        return None

# 👉 변경/교체: evaluate_mci_metric (정석 Cpk 계산)
def evaluate_mci_metric(
    df,
    variable,
    label,
    bundle,
    metadata,
    threshold=MCI_THRESHOLD,
    quantiles=MCI_QUANTILES,   # 참고용(규격으로 사용하지 않음)
    lsl_usl_map=None,          # {'var': {'LSL':..., 'USL':...}}
    use_within_sigma=True,
    within_group_size=5
):
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

    # 1) PDP 기반 safe range (정보 제공용/가상 규격 후보)
    pdp_payload, error = compute_pdp_curve(df, variable, metadata, bundle)
    if error:
        result["details"] = error
        return result

    safe_range = find_largest_safe_interval(
        pdp_payload["grid"],
        pdp_payload["averages"],
        threshold,
    )

    if safe_range is not None:
        result["safe_min"], result["safe_max"] = map(float, safe_range)

    # 2) 실제 공정 범위(참고용)
    series = pd.to_numeric(df[variable], errors="coerce").dropna()
    if series.empty:
        result["details"] = "실제 공정 데이터가 부족합니다."
        return result

    actual_min = float(series.min())
    actual_max = float(series.max())
    result["actual_min"] = actual_min
    result["actual_max"] = actual_max

    # 3) 평균과 σ
    mean = float(series.mean())
    if use_within_sigma:
        sigma = _estimate_within_sigma_via_rbar(series, within_group_size)
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            sigma = float(series.std(ddof=1))  # fallback: 전체 표본σ
    else:
        sigma = float(series.std(ddof=1))

    result["mean"] = mean
    result["std"] = sigma

    if not np.isfinite(sigma) or sigma <= 0:
        result["status"] = "위험"
        result["details"] = "표준편차를 계산할 수 없습니다."
        return result

    # 4) Cpk 규격 결정: LSL/USL 우선 → 없으면 safe range 사용(가상 규격, 해석 주의)
    LSL = USL = None
    if lsl_usl_map and variable in lsl_usl_map:
        LSL = lsl_usl_map[variable].get("LSL")
        USL = lsl_usl_map[variable].get("USL")

    if (LSL is None or USL is None) and safe_range is not None:
        LSL, USL = map(float, safe_range)

    if (LSL is None or USL is None) or not np.isfinite(LSL) or not np.isfinite(USL) or LSL >= USL:
        result["status"] = "데이터 부족"
        result["details"] = "Cpk 계산을 위한 LSL/USL이 없습니다. (엔지니어링 규격 제공 권장)"
        return result

    # 5) Cpk = min(Cpu, Cpl)  (절대값 사용 금지)
    cpu = (USL - mean) / (3 * sigma)
    cpl = (mean - LSL) / (3 * sigma)
    cpk = min(cpu, cpl)
    result["cpk"] = float(cpk)

    # 6) 상태 판정 (음수도 그대로 유지: 중심이 규격 밖일 수 있음)
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


# 👉 변경/교체: compute_mci_metrics (파라미터 전달 및 정석 Cpk)
def compute_mci_metrics(
    df,
    metadata,
    bundle,
    lsl_usl_map=None,
    use_within_sigma=True,
    within_group_size=5
):
    if bundle is None:
        return {"error": "모델 아티팩트를 찾을 수 없습니다.", "metrics": [], "overall_ratio": None}

    metrics = []
    for config in MCI_CONFIGS:
        metric = evaluate_mci_metric(
            df=df,
            variable=config["variable"],
            label=config["label"],
            bundle=bundle,
            metadata=metadata,
            threshold=config.get("threshold", MCI_THRESHOLD),
            lsl_usl_map=lsl_usl_map,
            use_within_sigma=use_within_sigma,
            within_group_size=within_group_size,
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
    df['date_only'] = df['registration_time'].dt.date
    
    date_stats = df.groupby('date_only').agg(
        defects=('passorfail', 'sum'),
        total=('passorfail', 'count')
    ).reset_index()
    
    date_stats['p'] = date_stats['defects'] / date_stats['total']
    
    p_bar = date_stats['defects'].sum() / date_stats['total'].sum()
    
    n_bar = date_stats['total'].mean()
    
    sigma = np.sqrt(p_bar * (1 - p_bar) / n_bar)
    
    UCL = p_bar + 3 * sigma
    LCL = p_bar - 3 * sigma
    LCL = max(0, LCL)
    
    return date_stats, p_bar, UCL, LCL

def calculate_p_chart(df, subgroup_size=5):
    if 'passorfail' not in df.columns:
        raise ValueError("passorfail 컬럼이 없습니다.")
    
    n_subgroups = len(df) // subgroup_size
    df_subgroups = df.head(n_subgroups * subgroup_size).copy()
    
    df_subgroups['subgroup'] = np.repeat(range(n_subgroups), subgroup_size)
    
    subgroup_stats = df_subgroups.groupby('subgroup').agg(
        defects=('passorfail', 'sum'),
        total=('passorfail', 'count')
    ).reset_index()
    
    subgroup_stats['p'] = subgroup_stats['defects'] / subgroup_stats['total']
    
    p_bar = subgroup_stats['defects'].sum() / subgroup_stats['total'].sum()
    
    n = subgroup_size
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    
    UCL = p_bar + 3 * sigma
    LCL = p_bar - 3 * sigma
    LCL = max(0, LCL)
    
    return subgroup_stats, p_bar, UCL, LCL

# Xbar-R 관리도 계산 함수 (날짜 기반 - registration_time 사용)
def calculate_xbar_r_chart_by_date(df, variable):
    """날짜 기반 Xbar-R 관리도 계산 (CSV의 평균/표준편차 사용)"""
    df['date_only'] = df['registration_time'].dt.date
    
    date_stats = df.groupby('date_only')[variable].agg([
        ('mean', 'mean'),
        ('range', lambda x: x.max() - x.min())
    ]).reset_index()
    
    # CSV 파일에서 관리한계 값 불러오기
    control_limits_df = load_control_limits()
    
    if control_limits_df is not None and 'mold_code' in df.columns:
        # 현재 데이터의 금형 코드 추출 (가장 많이 나온 금형 사용)
        mold_code = df['mold_code'].mode()[0] if not df['mold_code'].mode().empty else None
        
        if mold_code is not None:
            # CSV에서 해당 금형과 변수에 대한 평균, 표준편차, n 가져오기
            limit_row = control_limits_df[
                (control_limits_df['mold_code'] == int(mold_code)) & 
                (control_limits_df['variable'] == variable)
            ]
            
            if not limit_row.empty:
                xbar_bar = limit_row['mean'].values[0]
                std = limit_row['std'].values[0]
                n = limit_row['n'].values[0]
                
                # 표준편차 기반 관리한계 계산: X̄ ± 3σ/√n
                UCL_xbar = xbar_bar + 3 * std / np.sqrt(n)
                LCL_xbar = xbar_bar - 3 * std / np.sqrt(n)
                
                # R 관리도는 기존 방식 유지
                r_bar = date_stats['range'].mean()
                n_bar = df.groupby('date_only').size().mean()
                n_rounded = int(round(n_bar))
                n_rounded = max(2, min(10, n_rounded))
                
                control_chart_constants = {
                    2: {'D3': 0, 'D4': 3.267},
                    3: {'D3': 0, 'D4': 2.574},
                    4: {'D3': 0, 'D4': 2.282},
                    5: {'D3': 0, 'D4': 2.114},
                    6: {'D3': 0, 'D4': 2.004},
                    7: {'D3': 0.076, 'D4': 1.924},
                    8: {'D3': 0.136, 'D4': 1.864},
                    9: {'D3': 0.184, 'D4': 1.816},
                    10: {'D3': 0.223, 'D4': 1.777}
                }
                
                constants = control_chart_constants.get(n_rounded, control_chart_constants[5])
                D3 = constants['D3']
                D4 = constants['D4']
                
                UCL_r = D4 * r_bar
                LCL_r = D3 * r_bar
                
                return date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r
    
    # CSV 파일이 없거나 데이터를 찾을 수 없는 경우 기존 방식 사용
    xbar_bar = date_stats['mean'].mean()
    r_bar = date_stats['range'].mean()
    n_bar = df.groupby('date_only').size().mean()
    
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
    
    UCL_xbar = xbar_bar + A2 * r_bar
    LCL_xbar = xbar_bar - A2 * r_bar
    
    UCL_r = D4 * r_bar
    LCL_r = D3 * r_bar
    
    return date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r

# Xbar-R 관리도 계산 함수
def calculate_xbar_r_chart(df, variable, subgroup_size=5):
    """서브그룹 기반 Xbar-R 관리도 계산 (CSV의 평균/표준편차 사용)"""
    if variable not in df.columns:
        raise ValueError(f"{variable} 컬럼이 없습니다.")
    
    n_subgroups = len(df) // subgroup_size
    df_subgroups = df.head(n_subgroups * subgroup_size).copy()
    
    df_subgroups['subgroup'] = np.repeat(range(n_subgroups), subgroup_size)
    
    subgroup_stats = df_subgroups.groupby('subgroup')[variable].agg([
        ('mean', 'mean'),
        ('range', lambda x: x.max() - x.min())
    ]).reset_index()
    
    # CSV 파일에서 관리한계 값 불러오기
    control_limits_df = load_control_limits()
    
    if control_limits_df is not None and 'mold_code' in df.columns:
        # 현재 데이터의 금형 코드 추출 (가장 많이 나온 금형 사용)
        mold_code = df['mold_code'].mode()[0] if not df['mold_code'].mode().empty else None
        
        if mold_code is not None:
            # CSV에서 해당 금형과 변수에 대한 평균, 표준편차, n 가져오기
            limit_row = control_limits_df[
                (control_limits_df['mold_code'] == int(mold_code)) & 
                (control_limits_df['variable'] == variable)
            ]
            
            if not limit_row.empty:
                xbar_bar = limit_row['mean'].values[0]
                std = limit_row['std'].values[0]
                n = limit_row['n'].values[0]
                
                # 표준편차 기반 관리한계 계산: X̄ ± 3σ/√n
                UCL_xbar = xbar_bar + 3 * std / np.sqrt(n)
                LCL_xbar = xbar_bar - 3 * std / np.sqrt(n)
                
                # R 관리도는 기존 방식 유지
                r_bar = subgroup_stats['range'].mean()
                
                control_chart_constants = {
                    2: {'D3': 0, 'D4': 3.267},
                    3: {'D3': 0, 'D4': 2.574},
                    4: {'D3': 0, 'D4': 2.282},
                    5: {'D3': 0, 'D4': 2.114},
                    6: {'D3': 0, 'D4': 2.004},
                    7: {'D3': 0.076, 'D4': 1.924},
                    8: {'D3': 0.136, 'D4': 1.864},
                    9: {'D3': 0.184, 'D4': 1.816},
                    10: {'D3': 0.223, 'D4': 1.777}
                }
                
                if subgroup_size == 1:
                    constants = {'D3': 0, 'D4': 3.267}
                else:
                    constants = control_chart_constants.get(subgroup_size, control_chart_constants[5])
                
                D3 = constants['D3']
                D4 = constants['D4']
                
                UCL_r = D4 * r_bar
                LCL_r = D3 * r_bar
                
                return subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r
    
    # CSV 파일이 없거나 데이터를 찾을 수 없는 경우 기존 방식 사용
    xbar_bar = subgroup_stats['mean'].mean()
    r_bar = subgroup_stats['range'].mean()
    
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
    
    if subgroup_size == 1:
        moving_ranges = df_subgroups[variable].diff().abs()
        r_bar = moving_ranges.mean()
        constants = {'A2': 2.660, 'D3': 0, 'D4': 3.267}
    else:
        constants = control_chart_constants.get(subgroup_size, control_chart_constants[5])
    
    A2 = constants['A2']
    D3 = constants['D3']
    D4 = constants['D4']
    
    UCL_xbar = xbar_bar + A2 * r_bar
    LCL_xbar = xbar_bar - A2 * r_bar
    
    UCL_r = D4 * r_bar
    LCL_r = D3 * r_bar
    
    return subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r

# =========================
# 탭별 UI
# =========================
tab_ui = ui.page_fluid(
    ui.tags.style("""
        .mci-metric-card {
            position: relative;
            padding: 0;
            border-radius: 10px;
            min-height: 180px;
            background: #2A2D30;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            margin-bottom: 12px;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            overflow: hidden;
        }
        .mci-metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s ease; z-index: 10; pointer-events: none;
        }
        .mci-metric-card:hover { transform: translateY(-8px); box-shadow: 0 12px 32px rgba(0,0,0,0.25); }
        .mci-metric-card:hover::before { left: 100%; }
        .mci-metric-card.status-stable { border-left: 4px solid #28a745; }
        .mci-metric-card.status-caution { border-left: 4px solid #ffc107; }
        .mci-metric-card.status-danger { border-left: 4px solid #aeaeb1; }
        .mci-metric-card.status-unknown { border-left: 4px solid #6c757d; }
        .mci-card-outer { background: #2A2D30; border-radius: 10px; padding: 2.5px; height: 100%; display: flex; flex-direction: column; }
        .mci-card-header { color: white; padding: 8px; font-size: 0.8rem; font-weight: 600; text-align: center; letter-spacing: 0.5px; }
        .mci-card-inner { background: #f5f5f5; border-radius: 5px; padding: 24px 20px; flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .mci-value-main { font-size: 1.0rem; font-weight: 700; color: #000; margin-bottom: 8px; line-height: 1; animation: fadeInScale 0.6s ease-out; }
        @keyframes fadeInScale { from {opacity:0; transform:scale(0.8);} to {opacity:1; transform:scale(1);} }
        .mci-detail { font-size: 0.75rem; color: #555; margin: 3px 0; text-align: center; line-height: 1.3; animation: slideInUp 0.6s ease-out 0.1s both; }
        @keyframes slideInUp { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }
        .mci-cards-row { gap: 12px; margin-bottom: 20px; }
        .mci-info-icon { display: inline-block; width:16px; height:16px; background:#555; color:#fff; border-radius:50%; text-align:center; line-height:16px; font-size:0.7rem; font-weight:bold; cursor:pointer; margin-left:4px; transition: all .3s cubic-bezier(0.34,1.56,0.64,1); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .popover-body { white-space: pre-line; }
        .mci-info-icon:hover { background:#4A90E2; transform: scale(1.3) rotate(10deg); box-shadow: 0 6px 16px rgba(74,144,226,.5); }

        /* 대시보드 큰 카드 */
        .mci-dashboard-card{ --mci-radius:16px; border-radius:var(--mci-radius); overflow:hidden; }
        .mci-filter-header{ background:#2B2D30; color:#fff; padding:6px 12px; line-height:1.05; border-bottom:none; }
        .mci-section{ padding:12px 16px; }
        .mci-section-title{ font-weight:700; font-size:1rem; margin:8px 0 12px; }
        .mci-divider{ height:1px; background:rgba(0,0,0,.08); margin:8px 0 16px; }
        .mci-dashboard-card,
        .mci-dashboard-card * {
            font-family: inherit;
        }
        .mci-dashboard-card .form-label,
        .mci-dashboard-card label {
            font-size:0.85rem;
            font-weight:600;
            color:#2B2D30;
        }
        .mci-dashboard-card .selectize-input,
        .mci-dashboard-card .btn,
        .mci-dashboard-card .form-control,
        .mci-dashboard-card .form-select {
            font-size:0.85rem;
        }
        .mci-apply-btn{
            background-color:#2B2D30;
            border-color:#2B2D30;
            color:#ffffff;
            transition:background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }
        .mci-apply-btn:hover,
        .mci-apply-btn:focus{
            background-color:#35373B;
            border-color:#35373B;
            color:#ffffff;
            transform:translateY(-1px);
        }
        .mci-apply-btn:active{
            background-color:#1F2023;
            border-color:#1F2023;
            transform:translateY(0);
        }
        .mci-apply-btn:disabled{
            opacity:0.6;
            pointer-events:none;
        }
    """),

    # === 하나의 큰 카드 안에 모든 것(필터+개요+관리도) 포함 ===
    ui.div(ui.output_ui("mci_overview"), class_="mci-section"),
    ui.card(
        ui.card_header(
            ui.div(
                ui.span("필터 설정", class_="fw-semibold", style="font-size:0.9rem;"),
                ui.tags.button(
                    ui.span("접기", id="filter-toggle-label"),
                    id="filter-toggle-btn",
                    class_="btn btn-sm btn-outline-light",
                    type="button",
                    data_bs_toggle="collapse",
                    data_bs_target="#filter-section",
                    aria_expanded="true",
                    aria_controls="filter-section",
                    style="padding:2px 8px; font-size:0.75rem;"
                ),
                class_="d-flex justify-content-between align-items-center"
            ),
            class_="mci-filter-header"
        ),

        # (1) 필터 섹션 (접힘/펼침)
        ui.div(
            ui.div(
                ui.input_radio_buttons(
                    "analysis_mode", "분석 종류 선택",
                    choices={"subgroup":"서브그룹 기반 분석", "date":"날짜 기반 분석"},
                    selected="subgroup", inline=True, width="100%"
                ),
                ui.panel_conditional(
                    "input.analysis_mode === 'subgroup'",
                    ui.tags.p("현재 서브그룹 기반 분석은 n=100으로 고정되어 적용됩니다.", class_="text-muted small mb-2")
                ),
                class_="mb-3"
            ),
            ui.panel_conditional(
                "input.analysis_mode === 'date'",
                ui.div(
                    ui.input_date_range(
                        "date_range", "날짜 범위 (2019-01-02 ~ 2019-03-12)",
                        start="2019-01-02", end="2019-03-12", min="2019-01-02", max="2019-03-12", width="100%"
                    ),
                    class_="mb-3"
                ),
            ),
            ui.layout_columns(
                ui.input_selectize(
                    "mold_code", "금형 코드",
                    choices=MOLD_CODE_CHOICES, multiple=False,
                    selected="8600",
                    options={"placeholder": "금형 코드를 선택하세요"}
                ),
                ui.input_selectize(
                    "xbar_variable", "Xbar-R 변수",
                    choices=VARIABLE_LABELS, selected="sleeve_temperature",
                    options={"placeholder": "관리도에 사용할 변수를 선택하세요", "dropdownParent": "body"},
                    width="100%"
                ),
                col_widths=[6,6],
                class_="mb-3"
            ),
            ui.tags.p("선택한 변수 기준으로 P, Xbar, R 관리도가 계산됩니다.", class_="text-muted small mb-2"),
            ui.div(
                ui.input_action_button("reset_filter", "필터 초기화", class_="btn btn-outline-secondary me-2"),
                ui.input_action_button("apply_filter", "필터 적용", class_="btn mci-apply-btn"),
                class_="d-flex justify-content-end"
            ),
            id="filter-section",
            class_="collapse show mci-section"
        ),

        # 접힘/펼침 라벨 스크립트
        ui.tags.script("""
        (function(){
          function updateLabel(expanded){
            var l=document.getElementById('filter-toggle-label');
            var b=document.getElementById('filter-toggle-btn');
            if(!l||!b) return;
            l.textContent = expanded ? '접기' : '열기';
            b.setAttribute('aria-expanded', expanded ? 'true' : 'false');
          }
          function bind(){
            var s=document.getElementById('filter-section');
            if(!s||s.__bound) return; s.__bound=true;
            s.addEventListener('shown.bs.collapse',  function(){ updateLabel(true);  });
            s.addEventListener('hidden.bs.collapse', function(){ updateLabel(false); });
          }
          document.addEventListener('DOMContentLoaded', bind);
          document.addEventListener('shiny:connected', bind);
          document.addEventListener('shiny:value', function(){ setTimeout(bind, 0); });
        })();
        """),

        ui.div(class_="mci-divider"),

        # (2) MCI 개요 카드들

        # (3) P 관리도
        ui.div(
            ui.div("P 관리도 (불량률)", class_="mci-section-title"),
            ui.output_plot("plot_p_chart", height="550px"),
            class_="mci-section"
        ),

        # (4) Xbar / R 관리도
        ui.div(
            ui.layout_columns(
                ui.div(
                    ui.div("Xbar 관리도 (평균)", class_="mci-section-title"),
                    ui.output_plot("plot_xbar_chart", height="620px"),
                    class_="mci-section"
                ),
                ui.div(
                    ui.div("R 관리도 (범위)", class_="mci-section-title"),
                    ui.output_plot("plot_r_chart", height="620px"),
                    class_="mci-section"
                ),
                col_widths=[6,6]
            )
        ),

        class_="mci-dashboard-card"
    )
)

# =========================
# 탭별 서버
# =========================
def tab_server(input, output, session):
    # 필터 초기화 기능
    @reactive.Effect
    @reactive.event(input.reset_filter)
    def _():
        from shiny import ui as shiny_ui
        ui.update_radio_buttons("analysis_mode", selected="subgroup")
        ui.update_date_range("date_range", start="2019-01-02", end="2019-03-12")
        # 금형 코드를 8600으로 초기화
        ui.update_selectize(
            "mold_code", 
            choices=MOLD_CODE_CHOICES,
            selected="8600"
        )
        # Xbar-R 변수를 슬리브 온도로 초기화
        ui.update_selectize("xbar_variable", selected="sleeve_temperature")
    
    def current_filters():
        with reactive.isolate():
            analysis_mode = input.analysis_mode()
            mold_selection = input.mold_code()
            if isinstance(mold_selection, (list, tuple)):
                mold_codes = list(mold_selection)
            elif mold_selection:
                mold_codes = [mold_selection]
            else:
                mold_codes = []
            date_range = input.date_range()
            variable = input.xbar_variable()

        subgroup_size = 100
        subgroup_start = 0
        subgroup_end = 100

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

    # 👉 변경/교체: 정석 Cpk 파이프라인 사용 & within σ 옵션 전달
    @reactive.Calc
    def mci_metrics():
        input.apply_filter()
        bundle = get_model_bundle()
        df = mci_dataset()
        # 서브그룹 기반 군내 σ를 쓰고 싶으면 within_group_size를 현재 n(=100)과 맞춤
        return compute_mci_metrics(
            df=df,
            metadata=MODEL_METADATA,
            bundle=bundle,
            lsl_usl_map=LSL_USL_MAP,   # 규격 없으면 자동으로 safe range 사용(가상 규격)
            use_within_sigma=True,      # 군내 σ 권장
            within_group_size=100
        )

    @render.ui
    def mci_overview():
        metrics_payload = mci_metrics()
        error = metrics_payload.get("error")
        if error:
            return ui.card(ui.card_body(ui.p(error, class_="text-danger")), class_="mb-3")

        def format_cpk(value):
            if value is None:
                return "--"
            try:
                v_real = float(value)
            except Exception:
                return "--"

            v_show = abs(v_real) if DISPLAY_ABS_CPK else v_real
            # 표시 하한 적용(디스플레이 전용)
            v_show = max(v_show, CPK_DISPLAY_FLOOR)
            return f"{v_show:.2f}"


        def format_range(min_val, max_val):
            if min_val is None or max_val is None:
                return "데이터 없음"
            return f"[{min_val:.2f}, {max_val:.2f}]"
        
        def get_status_class(status):
            if status == "안정":
                return "stable"
            elif status == "주의":
                return "caution"
            elif status == "위험":
                return "danger"
            else:
                return "unknown"

        cards = []

        overall_cpk = metrics_payload.get("overall_cpk")
        overall_status = metrics_payload.get("overall_status", "데이터 부족")
        overall_status_class = get_status_class(overall_status)

        cards.append(
            ui.div(
                ui.div(
                    ui.div("모델 기반 최소 CPK", class_="mci-card-header"),
                    ui.div(
                        ui.div(
                            format_cpk(overall_cpk), 
                            class_="mci-value-main",
                            style="font-size: 2.2em; font-weight: 900;" if overall_status == "위험" else ""
                        ),
                        ui.p(
                            overall_status,
                            class_="mci-detail",
                            style="color: #dc3545; font-weight: 900; font-size: 1.4em;" if overall_status == "위험" else "",
                        ),
                        class_="mci-card-inner"
                    ),
                    class_="mci-card-outer"
                ),
                class_=f"mci-metric-card status-{overall_status_class}",
            )
        )

        for metric in metrics_payload.get("metrics", []):
            status = metric.get("status", "데이터 부족")
            status_class = get_status_class(status)

            safe_range_text = format_range(metric.get('safe_min'), metric.get('safe_max'))
            actual_range_text = format_range(metric.get('actual_min'), metric.get('actual_max'))
            tooltip_content = f"허용(모델/규격): {safe_range_text}\\n실제(참고): {actual_range_text}"

            cards.append(
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span(metric.get("label", metric.get("variable", "")), style="margin-right: 6px;"),
                            ui.span(
                                "ⓘ",
                                class_="mci-info-icon",
                                data_bs_toggle="tooltip",
                                data_bs_placement="right",
                                data_bs_trigger="click",
                                title=tooltip_content,
                                style="display: inline-block; cursor: pointer;"
                            ),
                            class_="mci-card-header"
                        ),
                        ui.div(
                            ui.div(
                                format_cpk(metric.get("cpk")), 
                                class_="mci-value-main",
                                style="font-size: 2.2em; font-weight: 900;" if status == "위험" else ""
                            ),
                            ui.p(
                                ui.span(
                                    status,
                                    style="color: #dc3545; font-weight: 900; font-size: 1.4em;" if status == "위험" else "",
                                ),
                                class_="mci-detail"
                            ),
                            class_="mci-card-inner"
                        ),
                        class_="mci-card-outer"
                    ),
                    class_=f"mci-metric-card status-{status_class}",
                )
            )

        return ui.div(
            ui.layout_columns(*cards, col_widths=[3] * len(cards)),
            class_="mci-cards-row",
        )

    @render.plot
    def plot_p_chart():
        input.apply_filter()
        filters = current_filters()
        mold_codes = filters["mold_codes"] or None
        analysis_mode = filters["analysis_mode"]

        if analysis_mode == "date":
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            date_stats, p_bar, UCL, LCL = calculate_p_chart_by_date(df)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = 'P 관리도 (날짜 기반)'
        else:
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            subgroup_stats, p_bar, UCL, LCL = calculate_p_chart(df, subgroup_size=subgroup_size)
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'P 관리도 (서브그룹 기반, n={subgroup_size})'

        fig, ax = plt.subplots(figsize=(14, 6))

        if len(display_data) > 0 and analysis_mode == "subgroup":
            ax.set_xlim(display_data[x_column].min() - 1, display_data[x_column].max() + 1)

        warn_upper = p_bar + (UCL - p_bar) * 2/3
        warn_lower = p_bar - (p_bar - LCL) * 2/3
        out_of_control = (display_data['p'] > UCL) | (display_data['p'] < LCL)

        ax.plot(display_data[x_column], display_data['p'], color='#2E86AB', linewidth=2, linestyle='-', zorder=2)
        ax.scatter(display_data.loc[~out_of_control, x_column], display_data.loc[~out_of_control, 'p'],
                   color='#2E86AB', s=45, marker='o', label='불량률 (P)', zorder=3)
        if out_of_control.any():
            ax.scatter(display_data.loc[out_of_control, x_column], display_data.loc[out_of_control, 'p'],
                       color='red', s=55, marker='o', label='관리한계 초과', zorder=4)

        ax.axhline(y=p_bar, color='green', linestyle='-', linewidth=2, label=f'중심선 (P̄ = {p_bar:.4f})', zorder=1)
        ax.axhline(y=UCL, color='red', linestyle='--', linewidth=2, label=f'UCL = {UCL:.4f}', zorder=1)
        ax.axhline(y=LCL, color='red', linestyle='--', linewidth=2, label=f'LCL = {LCL:.4f}', zorder=1)
        ax.axhline(y=warn_upper, color='orange', linestyle=':', linewidth=1.5, label=f'+2σ = {warn_upper:.4f}', zorder=1)
        ax.axhline(y=warn_lower, color='orange', linestyle=':', linewidth=1.5, label=f'-2σ = {warn_lower:.4f}', zorder=1)

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('불량률 (P)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85, borderpad=0.6, labelspacing=0.4)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    @render.plot
    def plot_xbar_chart():
        input.apply_filter()
        filters = current_filters()

        analysis_mode = filters["analysis_mode"]
        variable = filters["variable"]
        variable_label = VARIABLE_LABELS.get(variable, variable)
        mold_codes = filters["mold_codes"] or None

        if analysis_mode == "date":
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = 'Xbar 관리도 (날짜 기반)'
        else:
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'Xbar 관리도 (서브그룹 기반, n={subgroup_size})'

        fig, ax = plt.subplots(figsize=(14, 6))

        warn_upper_xbar = xbar_bar + (UCL_xbar - xbar_bar) * 2/3
        warn_lower_xbar = xbar_bar - (xbar_bar - LCL_xbar) * 2/3
        out_of_control_xbar = (display_data['mean'] > UCL_xbar) | (display_data['mean'] < LCL_xbar)

        ax.plot(display_data[x_column], display_data['mean'], color='#2E86AB', linewidth=2, linestyle='-', zorder=2)
        ax.scatter(display_data.loc[~out_of_control_xbar, x_column], display_data.loc[~out_of_control_xbar, 'mean'],
                   color='#2E86AB', s=45, marker='o', label='Xbar (평균)', zorder=3)
        if out_of_control_xbar.any():
            ax.scatter(display_data.loc[out_of_control_xbar, x_column], display_data.loc[out_of_control_xbar, 'mean'],
                       color='red', s=55, marker='o', label='관리한계 초과', zorder=4)

        ax.axhline(y=xbar_bar, color='green', linestyle='-', linewidth=2, label=f'X̿ = {xbar_bar:.4f}', zorder=1)
        ax.axhline(y=UCL_xbar, color='red', linestyle='--', linewidth=2, label=f'UCL = {UCL_xbar:.4f}', zorder=1)
        ax.axhline(y=LCL_xbar, color='red', linestyle='--', linewidth=2, label=f'LCL = {LCL_xbar:.4f}', zorder=1)
        ax.axhline(y=warn_upper_xbar, color='orange', linestyle=':', linewidth=1.5, label=f'+2σ = {warn_upper_xbar:.4f}', zorder=1)
        ax.axhline(y=warn_lower_xbar, color='orange', linestyle=':', linewidth=1.5, label=f'-2σ = {warn_lower_xbar:.4f}', zorder=1)

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable_label} - 평균', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False, handlelength=2.5, columnspacing=1.5)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.28)
        return fig

    @render.plot
    def plot_r_chart():
        input.apply_filter()
        filters = current_filters()

        analysis_mode = filters["analysis_mode"]
        variable = filters["variable"]
        variable_label = VARIABLE_LABELS.get(variable, variable)
        mold_codes = filters["mold_codes"] or None

        if analysis_mode == "date":
            date_range = filters["date_range"]
            date_start = date_range[0] if date_range else None
            date_end = date_range[1] if date_range else None
            df = load_and_filter_data(date_start=date_start, date_end=date_end, mold_codes=mold_codes)
            date_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = calculate_xbar_r_chart_by_date(df, variable)
            display_data = date_stats
            x_column = 'date_only'
            x_label = '날짜'
            title = 'R 관리도 (날짜 기반)'
        else:
            df = load_and_filter_data(mold_codes=mold_codes)
            subgroup_size = filters["subgroup_size"]
            subgroup_stats, xbar_bar, r_bar, UCL_xbar, LCL_xbar, UCL_r, LCL_r = calculate_xbar_r_chart(df, variable, subgroup_size=subgroup_size)
            start = max(0, filters["subgroup_start"])
            end = min(len(subgroup_stats), filters["subgroup_end"])
            display_data = subgroup_stats.iloc[start:end]
            x_column = 'subgroup'
            x_label = '서브그룹 번호'
            title = f'R 관리도 (서브그룹 기반, n={subgroup_size})'

        fig, ax = plt.subplots(figsize=(14, 6))

        warn_upper_r = r_bar + (UCL_r - r_bar) * 2/3
        warn_lower_r = r_bar - (r_bar - LCL_r) * 2/3
        out_of_control_r = (display_data['range'] > UCL_r) | (display_data['range'] < LCL_r)

        ax.plot(display_data[x_column], display_data['range'], color="#2E86AB", linewidth=2, linestyle='-', zorder=2)
        ax.scatter(display_data.loc[~out_of_control_r, x_column], display_data.loc[~out_of_control_r, 'range'],
                   color='#2E86AB', s=45, marker='o', label='R (범위)', zorder=3)
        if out_of_control_r.any():
            ax.scatter(display_data.loc[out_of_control_r, x_column], display_data.loc[out_of_control_r, 'range'],
                       color='red', s=55, marker='o', label='관리한계 초과', zorder=4)

        ax.axhline(y=r_bar, color='green', linestyle='-', linewidth=2, label=f'R̄ = {r_bar:.4f}', zorder=1)
        ax.axhline(y=UCL_r, color='red', linestyle='--', linewidth=2, label=f'UCL = {UCL_r:.4f}', zorder=1)
        ax.axhline(y=LCL_r, color='red', linestyle='--', linewidth=2, label=f'LCL = {LCL_r:.4f}', zorder=1)
        ax.axhline(y=warn_upper_r, color='orange', linestyle=':', linewidth=1.5, label=f'+2σ = {warn_upper_r:.4f}', zorder=1)
        ax.axhline(y=warn_lower_r, color='orange', linestyle=':', linewidth=1.5, label=f'-2σ = {warn_lower_r:.4f}', zorder=1)

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable_label} - 범위', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False, handlelength=2.5, columnspacing=1.5)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        if analysis_mode == "date":
            ax.tick_params(axis='x', rotation=45)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.28)
        return fig
