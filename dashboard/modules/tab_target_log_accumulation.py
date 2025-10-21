from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 한글 폰트 설정 ==========
def setup_korean_font():
    """유니코드 마이너스를 포함한 한글 폰트 설정 (DejaVu Sans 폴백)"""
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    korean_fonts = [
        'Noto Sans KR',
        'Noto Sans CJK KR',
        'NanumGothic',
        'AppleGothic',
        'Malgun Gothic'
    ]

    chosen = next((f for f in korean_fonts if f in available_fonts), None)

    if chosen:
        plt.rcParams['font.family'] = [chosen, 'DejaVu Sans']
        print(f"✓ 폰트 설정: {chosen} + DejaVu Sans (fallback)")
    else:
        plt.rcParams['font.family'] = ['DejaVu Sans']
        print("✓ 폰트 설정: DejaVu Sans")

    plt.rcParams['axes.unicode_minus'] = True

setup_korean_font()

# 경로 및 상수
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"
DROP_COLUMNS = ['line', 'name', 'mold_name', 'date', 'time', 'Unnamed: 0', 'id']

COLUMN_NAMES_KR = {
    "registration_time": "등록 일시", "count": "생산 순번", "working": "가동 여부",
    "emergency_stop": "비상 정지", "facility_operation_cycleTime": "설비 운영 사이클타임",
    "production_cycletime": "제품 생산 사이클타임", "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도", "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스킷 두께", "upper_mold_temp1": "상부 금형 온도1",
    "upper_mold_temp2": "상부 금형 온도2", "upper_mold_temp3": "상부 금형 온도3",
    "lower_mold_temp1": "하부 금형 온도1", "lower_mold_temp2": "하부 금형 온도2",
    "lower_mold_temp3": "하부 금형 온도3", "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도", "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "전자교반 가동시간", "mold_code": "금형 코드",
    "tryshot_signal": "트라이샷 신호", "molten_temp": "용탕 온도",
    "molten_volume": "용탕 부피", "heating_furnace": "가열로",
    "passorfail": "불량 여부", "uniformity": "균일도",
    "mold_temp_udiff": "금형 온도차(상/하)", "P_diff": "압력 차이",
    "Cycle_diff": "사이클 시간 차이"
}

# ========== 모델 로드 (임계값 수정) ==========
try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    ordinal_encoder = artifact.get("ordinal_encoder")
    onehot_encoder = artifact.get("onehot_encoder")
    
    # 임계값 확인 및 설정
    threshold = artifact.get("operating_threshold")
    if threshold is None:
        threshold = 0.7553
        print(f"⚠️ 모델 파일에 임계값 없음 → 0.7553로 설정")
    else:
        threshold = float(threshold)
        print(f"✓ 모델 임계값: {threshold}")
    if abs(threshold - 0.5) < 0.001:
        print(f"⚠️ 임계값이 0.5로 설정되어 있음 → 0.7553으로 변경")
        threshold = 0.7553

    # ===== 핵심 분리: 모델용 스키마 vs UI/추천용 스키마 =====
    model_numeric_cols = list(scaler.feature_names_in_) if scaler and hasattr(scaler, 'feature_names_in_') else []
    model_categorical_cols = list(ordinal_encoder.feature_names_in_) if ordinal_encoder and hasattr(ordinal_encoder, 'feature_names_in_') else []
    model_required_cols = model_numeric_cols + model_categorical_cols

    # UI/추천에서만 손댈 수 있는 복사본 (모델 예측 경로에는 절대 사용 X)
    ui_numeric_cols = model_numeric_cols.copy()
    ui_categorical_cols = model_categorical_cols.copy()

    # SHAP 매핑(모델 기준)
    explainer = shap.TreeExplainer(model)
    numeric_index_map = {feat: idx for idx, feat in enumerate(model_numeric_cols)}
    ohe_feature_slices = {}
    start_idx = len(model_numeric_cols)
    if model_categorical_cols and onehot_encoder is not None:
        for feat, ohe_cats in zip(model_categorical_cols, onehot_encoder.categories_):
            length = len(ohe_cats)
            ohe_feature_slices[feat] = (start_idx, start_idx + length)
            start_idx += length

    NUMERIC_FEATURE_RANGES = {}  # UI용 범위
    input_metadata = {}          # UI용 메타
    MODEL_LOADED = True

    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    MODEL_LOADED = False
    model = None
    explainer = None
    model_required_cols = []
    model_numeric_cols = []
    model_categorical_cols = []
    ui_numeric_cols = []
    ui_categorical_cols = []
    NUMERIC_FEATURE_RANGES = {}
    input_metadata = {}
    threshold = 0.7553  # 기본값

def _booster_iteration(model):
    iteration = getattr(model, "best_iteration", None)
    if not iteration:
        current_iteration = getattr(model, "current_iteration", None)
        if callable(current_iteration):
            iteration = current_iteration()
    return int(iteration) if iteration else None

def format_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)

# ===================== 예측 경로(모델 스키마만 사용) =====================

def build_input_dataframe(row_dict):
    data = {col: row_dict.get(col) for col in model_required_cols}
    input_df = pd.DataFrame([data], columns=model_required_cols)
    if model_numeric_cols:
        input_df[model_numeric_cols] = input_df[model_numeric_cols].apply(pd.to_numeric, errors="coerce")
    if model_categorical_cols:
        input_df[model_categorical_cols] = input_df[model_categorical_cols].fillna("UNKNOWN").astype(str)
    return input_df

def prepare_feature_matrix(input_df):
    if not MODEL_LOADED:
        return None
    try:
        arrays = []
        if model_numeric_cols and scaler is not None:
            arrays.append(scaler.transform(input_df[model_numeric_cols].astype(float)))
        if model_categorical_cols and ordinal_encoder is not None and onehot_encoder is not None:
            cat_ord = ordinal_encoder.transform(input_df[model_categorical_cols]).astype(int)
            cat_ohe = onehot_encoder.transform(cat_ord)
            if hasattr(cat_ohe, "toarray"):
                cat_ohe = cat_ohe.toarray()
            arrays.append(cat_ohe)
        return np.hstack(arrays).astype(np.float32) if arrays else np.zeros((len(input_df), 0), dtype=np.float32)
    except Exception as e:
        print(f"[prepare_feature_matrix] error: {e}")
        import traceback; traceback.print_exc()
        return None

def predict_passorfail(df):
    if not MODEL_LOADED or df.empty:
        return np.zeros(len(df), dtype=int)
    try:
        X = df.drop(columns=DROP_COLUMNS + ['passorfail'], errors='ignore').copy()

        # 모델 스키마에 맞춰 결측 채움
        for col in model_required_cols:
            if col not in X.columns:
                X[col] = 0.0 if col in model_numeric_cols else 'UNKNOWN'
        X = X[model_required_cols].copy()

        # 변환
        if model_numeric_cols and scaler is not None:
            X_num = X[model_numeric_cols].fillna(0.0)
            X_num_scaled = scaler.transform(X_num)
        else:
            X_num_scaled = np.empty((len(X), 0))

        if model_categorical_cols and ordinal_encoder is not None and onehot_encoder is not None:
            X_cat = X[model_categorical_cols].fillna('UNKNOWN')
            X_cat_ord = ordinal_encoder.transform(X_cat).astype(int)
            X_cat_ohe = onehot_encoder.transform(X_cat_ord)
        else:
            X_cat_ohe = np.empty((len(X), 0))

        if X_num_scaled.size and X_cat_ohe.size:
            X_final = np.hstack([X_num_scaled, X_cat_ohe])
        elif X_num_scaled.size:
            X_final = X_num_scaled
        elif X_cat_ohe.size:
            X_final = X_cat_ohe
        else:
            return np.zeros(len(df), dtype=int)

        iteration = _booster_iteration(model)
        probs = model.predict(X_final, num_iteration=iteration) if iteration else model.predict(X_final)
        predictions = (probs >= threshold).astype(int)

        # tryshot 규칙(실사용 판단)
        if 'tryshot_signal' in df.columns:
            predictions[df['tryshot_signal'].to_numpy() == 'D'] = 1

        return predictions
    except Exception as e:
        print(f"❌ predict_passorfail 실패: {e}")
        import traceback; traceback.print_exc()
        return np.zeros(len(df), dtype=int)

# 탐색용/실사용용 스위치
def evaluate_prediction(row_dict, *, ignore_tryshot=False, use_threshold=None):
    """단일 예측 (탐색 시 tryshot 무시 가능, 임계값 임시 지정 가능)"""
    if not MODEL_LOADED:
        return 0, 0.0
    try:
        input_df = build_input_dataframe(row_dict)
        feature_matrix = prepare_feature_matrix(input_df)
        if feature_matrix is None:
            return 0, 0.0

        iteration = _booster_iteration(model)
        probs = model.predict(feature_matrix, num_iteration=iteration) if iteration else model.predict(feature_matrix)
        probability = float(probs[0])

        thr = threshold if use_threshold is None else float(use_threshold)
        prediction = 1 if probability >= thr else 0

        if not ignore_tryshot:
            tryshot = row_dict.get("tryshot_signal")
            if tryshot is not None and str(tryshot).upper() == "D":
                prediction = 1

        return prediction, probability
    except Exception as e:
        print(f"[evaluate_prediction] error: {e}")
        import traceback; traceback.print_exc()
        return 0, 0.0

def evaluate_prediction_strict(row_dict):
    return evaluate_prediction(row_dict, ignore_tryshot=False)

def evaluate_prediction_for_search(row_dict):
    return evaluate_prediction(row_dict, ignore_tryshot=True)

# ===================== SHAP (모델 스키마 기준) =====================

def compute_shap_contributions(feature_matrix):
    if not MODEL_LOADED or explainer is None or feature_matrix is None:
        return None, None
    try:
        shap_values = explainer.shap_values(feature_matrix)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_vector = shap_values[0]
        contributions = {}

        for feat, idx in numeric_index_map.items():
            contributions[feat] = float(shap_vector[idx])

        for feat, (start, end) in ohe_feature_slices.items():
            contributions[feat] = float(np.sum(shap_vector[start:end])) if end > start else 0.0

        return contributions, shap_vector
    except Exception as e:
        print(f"[compute_shap_contributions] error: {e}")
        import traceback; traceback.print_exc()
        return None, None

def _extract_feature_values(row_dict):
    values = []
    for col in model_required_cols:
        val = row_dict.get(col, np.nan)
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else np.nan
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            val = str(val)
        values.append(np.nan if pd.isna(val) else val)
    return values

def build_shap_explanation(contributions, shap_vector, input_row):
    if contributions is None or shap_vector is None:
        return None
    try:
        from shap import Explanation
        expected_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

        shap_values = np.array([float(contributions.get(col, 0.0)) for col in model_required_cols], dtype=float)
        feature_values = np.array(_extract_feature_values(input_row), dtype=object)
        feature_names = [COLUMN_NAMES_KR.get(col, col) for col in model_required_cols]

        return Explanation(values=shap_values, base_values=expected_value, data=feature_values, feature_names=feature_names)
    except Exception as e:
        print(f"[build_shap_explanation] error: {e}")
        import traceback; traceback.print_exc()
        return None

# ===================== UI/추천 로직 (ui_*만 사용) =====================

def update_numeric_feature_ranges(df):
    """D) 1~99 퍼센타일 + ±5% 버퍼로 안전하게 범위 업데이트 (UI용)"""
    global NUMERIC_FEATURE_RANGES
    if df.empty:
        print("⚠️ 빈 데이터프레임 - 범위 업데이트 건너뜀")
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"🔧 NUMERIC_FEATURE_RANGES 업데이트 시작")
        print(f"{'='*60}")
        print(f"전체 데이터: {len(df)}행")
        
        if "passorfail" in df.columns:
            pass_df = df[df["passorfail"] == 0].copy()
            print(f"✓ 1단계: passorfail 컬럼 존재")
            print(f"  - 정상 데이터(0): {len(pass_df)}행")
            print(f"  - 불량 데이터(1): {len(df) - len(pass_df)}행")
        else:
            pass_df = pd.DataFrame()
            print(f"⚠️ 1단계: passorfail 컬럼 없음")
        
        if pass_df.empty:
            pass_df = df.copy()
            print(f"⚠️ 2단계: 정상 데이터 없음 → 전체 데이터 사용 ({len(pass_df)}행)")
        
        NUMERIC_FEATURE_RANGES = {}
        success_cols, failed_cols = [], []
        
        for col in ui_numeric_cols:
            if col not in pass_df.columns:
                failed_cols.append(f"{col} (컬럼 없음)")
                continue
            series = pd.to_numeric(pass_df[col], errors='coerce').dropna()
            if series.empty and col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                failed_cols.append(f"{col} (변환 실패)")
                continue

            q_low, q_high = series.quantile(0.01), series.quantile(0.99)
            span = (q_high - q_low)
            if not np.isfinite(span) or span <= 0:
                if q_low != 0:
                    min_val = q_low * 0.9
                    max_val = q_low * 1.1
                else:
                    min_val, max_val = -1.0, 1.0
            else:
                min_val = float(q_low - 0.05 * span)
                max_val = float(q_high + 0.05 * span)

            NUMERIC_FEATURE_RANGES[col] = (min_val, max_val)
            success_cols.append(col)
        
        if not NUMERIC_FEATURE_RANGES:
            print(f"❌ 3단계: 모든 변수 변환 실패 - 기본 범위 사용")
            for col in ui_numeric_cols:
                NUMERIC_FEATURE_RANGES[col] = (0.0, 100.0)
            print(f"✓ 기본 범위 설정: {len(NUMERIC_FEATURE_RANGES)}개 변수")
        else:
            print(f"\n✅ 성공: {len(success_cols)}개 변수")
            for col in success_cols[:5]:
                min_v, max_v = NUMERIC_FEATURE_RANGES[col]
                print(f"  {col}: [{min_v:.4f}, {max_v:.4f}]")
            if len(success_cols) > 5:
                print(f"  ... 외 {len(success_cols) - 5}개")
        
        if failed_cols:
            print(f"\n⚠️ 실패: {len(failed_cols)}개")
            for fail in failed_cols[:3]:
                print(f"  {fail}")
            if len(failed_cols) > 3:
                print(f"  ... 외 {len(failed_cols) - 3}개")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ 범위 업데이트 중 에러: {e}")
        import traceback; traceback.print_exc()
        if not NUMERIC_FEATURE_RANGES:
            print(f"⚠️ 에러 복구: 기본 범위 설정")
            for col in ui_numeric_cols:
                NUMERIC_FEATURE_RANGES[col] = (0.0, 100.0)
            print(f"✓ 기본 범위: {len(NUMERIC_FEATURE_RANGES)}개")

def create_input_metadata(df):
    """C) 입력 메타데이터 생성 (범주형 보정 + 수치형 퍼센타일 범위) - UI 스키마만 수정"""
    global input_metadata, ui_numeric_cols, ui_categorical_cols
    
    metadata = {}

    # 0) 휴리스틱 범주형 강제 편입 (코드/ID/플래그류)
    heuristic_categorical = []
    for col in df.columns:
        col_l = str(col).lower()
        if col_l in ["mold_code", "mold", "code", "model_code"]:
            heuristic_categorical.append(col)
        # 정수형이면서 유니크가 적은 플래그성 변수
        if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique(dropna=True) <= 20:
            heuristic_categorical.append(col)
    heuristic_categorical = list(dict.fromkeys(heuristic_categorical))  # unique

    # 기존 모델 스키마를 기반으로 UI 스키마 보정
    cat_set = set(ui_categorical_cols) | set(heuristic_categorical)
    num_set = set(ui_numeric_cols) - set(heuristic_categorical)

    # 1) 범주형 메타
    for col in cat_set:
        if col not in df.columns:
            continue
        values = sorted([str(v) for v in df[col].astype(str).dropna().unique()])
        if values:
            metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}

    # 2) 수치형 메타 (1~99퍼센타일)
    for col in num_set:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            continue
        vmin, vmax = float(s.quantile(0.01)), float(s.quantile(0.99))
        if vmin == vmax:
            vmin -= 1.0; vmax += 1.0
        metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": float(s.median())}

    # UI 스키마 갱신(모델 스키마에는 영향 없음)
    ui_categorical_cols = list(cat_set)
    ui_numeric_cols = list(num_set)
    input_metadata = metadata

    print(f"✓ 입력 메타데이터 생성: 범주형 {len([m for m in metadata.values() if m['type'] == 'categorical'])}개, 수치형 {len([m for m in metadata.values() if m['type'] == 'numeric'])}개")

# ===================== 추천 로직 (탐색 중 tryshot 무시 + 최소 확률 백업) =====================

def find_normal_range_binary_fixed(base_row, feature, bounds, tol_ratio=0.01, max_iter=15, n_check=5):
    """단일 변수 이진 탐색으로 정상 범위 찾기(+최소 확률 백업)"""
    if not bounds:
        return None
    f_min, f_max = bounds
    if pd.isna(f_min) or pd.isna(f_max) or f_min >= f_max:
        return None
    
    low, high = float(f_min), float(f_max)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    
    tol = max((high - low) * tol_ratio, 1e-3)
    best_details = None
    best_overall = None  # 정상 불가 시 최소 확률 (val, prob)

    for _ in range(max_iter):
        samples = np.linspace(low, high, n_check)
        normal_samples = []

        for val in samples:
            trial = base_row.copy()
            trial[feature] = float(val)
            pred, prob = evaluate_prediction_for_search(trial)

            if (best_overall is None) or (prob < best_overall[1]):
                best_overall = (float(val), float(prob))
            if pred == 0:
                normal_samples.append((float(val), float(prob)))

        if not normal_samples:
            break

        normal_samples.sort(key=lambda x: x[1])
        low = min(v for v, _ in normal_samples)
        high = max(v for v, _ in normal_samples)
        top_val, top_prob = normal_samples[0]

        examples = [top_val]
        if low not in examples: examples.append(low)
        if high not in examples: examples.append(high)

        if (best_details is None) or (top_prob < best_details[3]):
            best_details = (low, high, examples[:3], top_prob)

        if (high - low) <= tol:
            break

    if best_details is None:
        if best_overall is not None:
            return {
                "min": float(bounds[0]),
                "max": float(bounds[1]),
                "examples": [float(best_overall[0])],
                "best_prob": float(best_overall[1]),
                "status": "no-normal-but-best"
            }
        return None
    
    low, high, examples, best_prob = best_details
    return {
        "min": float(low),
        "max": float(high),
        "examples": [float(v) for v in examples],
        "best_prob": float(best_prob),
        "status": "normal-found"
    }

def binary_search_normal_ranges(base_row, features, feature_ranges, tol_ratio=0.01, max_iter=10):
    """다변수 동시 이진 탐색(+정상 불가 시 최소 확률 조합 백업)"""
    usable = {}
    for feat in features:
        bounds = feature_ranges.get(feat)
        if not bounds:
            continue
        f_min, f_max = bounds
        if pd.isna(f_min) or pd.isna(f_max) or not np.isfinite(f_min) or not np.isfinite(f_max):
            continue
        if f_min >= f_max:
            continue
        usable[feat] = [float(f_min), float(f_max)]
    
    if not usable:
        return None, {}, None, "no-features"
    
    print(f"\n🔄 다변수 동시 최적화 시작: {list(usable.keys())}")
    
    best_solution = None
    best_prob = None
    best_any_solution = None
    best_any_prob = None

    status = "normal-found"

    for iteration in range(max_iter):
        trial = base_row.copy()
        mids = {}
        for feat, (low, high) in usable.items():
            mid = (low + high) / 2.0
            mids[feat] = mid
            trial[feat] = mid
        
        pred, prob = evaluate_prediction_for_search(trial)
        is_normal = pred == 0

        if (best_any_prob is None) or (prob < best_any_prob):
            best_any_prob = float(prob)
            best_any_solution = {feat: float(val) for feat, val in mids.items()}

        if is_normal:
            if (best_prob is None) or (prob < best_prob):
                best_prob = float(prob)
                best_solution = {feat: float(val) for feat, val in mids.items()}
                print(f"  반복 {iteration+1}: 정상 발견! prob={best_prob:.4f}, 조합={best_solution}")
        else:
            status = "no-normal-but-best"

        updated = False
        for feat, (low, high) in list(usable.items()):
            mid = mids[feat]
            left = mid - low
            right = high - mid

            base_range = feature_ranges[feat]
            tol = max((base_range[1] - base_range[0]) * tol_ratio, 1e-3)

            if is_normal:
                new_range = [low, mid] if left >= right else [mid, high]
            else:
                new_range = [mid, high] if left >= right else [low, mid]

            if abs(new_range[1] - new_range[0]) < tol:
                new_range = [float(new_range[0]), float(new_range[1])]

            if new_range != usable[feat]:
                usable[feat] = new_range
                updated = True
        
        if not updated:
            break
    
    if best_solution is None and best_any_solution is not None:
        print(f"✗ 다변수 최적화: 정상 조합 없음 → 최소 확률 조합 반환 prob={best_any_prob:.4f}")
        return best_any_solution, {feat: tuple(bounds) for feat, bounds in usable.items()}, best_any_prob, "no-normal-but-best"

    if best_solution:
        print(f"✅ 다변수 최적화 성공: {len(best_solution)}개 변수, 최종 prob={best_prob:.4f}")
        return best_solution, {feat: tuple(bounds) for feat, bounds in usable.items()}, best_prob, "normal-found"

    print(f"✗ 다변수 최적화 실패: 조합 없음")
    return None, {feat: tuple(bounds) for feat, bounds in usable.items()}, None, "no-solution"

def evaluate_categorical_candidates(base_row, feature, choices, top_k=3):
    """범주형 변수의 정상 후보 찾기 (tryshot 무시)"""
    print(f"\n🔍 범주형 평가: {feature}, 후보 {len(choices)}개")
    candidates = []
    best_any = None

    for value in choices:
        trial = base_row.copy()
        trial[feature] = value
        pred, prob = evaluate_prediction_for_search(trial)

        if (best_any is None) or (prob < best_any[1]):
            best_any = (value, float(prob))

        if pred == 0:
            candidates.append((value, float(prob)))
            print(f"  ✓ {value}: prob={prob:.4f} (정상)")

    if candidates:
        candidates.sort(key=lambda x: x[1])
        values = [val for val, _ in candidates[:top_k]]
        print(f"  ✅ 추천 값(정상 전환): {values[:top_k]}")
        return {"values": values, "best_prob": float(candidates[0][1]), "status": "normal-found"}

    print(f"  ✗ 정상 전환 가능한 값 없음 → 최소 확률 후보 반환")
    return {"values": [best_any[0]] if best_any else [], "best_prob": float(best_any[1]) if best_any else None, "status": "no-normal-but-best"}

def recommend_ranges(base_row, focus_features):
    """3단계 전략으로 추천 구간 생성 (A,B,C,D 반영) - UI 스키마 기준"""
    if not focus_features:
        return {}
    
    print(f"\n{'='*60}")
    print(f"🎯 추천 구간 계산 시작")
    print(f"{'='*60}")
    print(f"대상 변수: {focus_features}")
    
    recommendations = {}
    best_prob = None
    
    numeric_targets = [feat for feat in focus_features if feat in ui_numeric_cols]
    categorical_targets = [feat for feat in focus_features if feat in ui_categorical_cols]
    
    print(f"\n분류: 수치형 {len(numeric_targets)}개, 범주형 {len(categorical_targets)}개")
    
    numeric_ranges = {}
    for feat in numeric_targets:
        bounds = NUMERIC_FEATURE_RANGES.get(feat)
        if bounds:
            numeric_ranges[feat] = bounds
    
    # 1) 다변수 동시 최적화
    multi_status = None
    if len(numeric_ranges) >= 2:
        print(f"\n--- 1단계: 다변수 동시 최적화 ---")
        solution, final_ranges, prob_multi, multi_status = binary_search_normal_ranges(
            base_row,
            list(numeric_ranges.keys()),
            numeric_ranges
        )
        if solution:
            for feat, mid in solution.items():
                bounds = final_ranges.get(feat, numeric_ranges.get(feat))
                if not bounds:
                    continue
                record = recommendations.get(feat, {"type": "numeric"})
                record["min"] = float(bounds[0])
                record["max"] = float(bounds[1])
                examples = record.get("examples", [])
                if mid not in examples: examples.append(float(mid))
                record["examples"] = examples[:3]
                record["method"] = "binary_multi"
                record["status"] = multi_status or "normal-found"
                recommendations[feat] = record
            if prob_multi is not None:
                best_prob = prob_multi if best_prob is None else min(best_prob, prob_multi)
    
    # 2) 단일 변수 탐색
    print(f"\n--- 2단계: 단일 변수 이진 탐색 ---")
    for feat, bounds in numeric_ranges.items():
        print(f"\n탐색: {feat}")
        details = find_normal_range_binary_fixed(base_row, feat, bounds)
        if not details:
            print(f"  ✗ {feat}: 정상 범위 및 최소 확률 추천 없음")
            continue
        print(f"  • {feat}: [{details['min']:.4f}, {details['max']:.4f}], prob={details.get('best_prob', float('nan')):.4f}, status={details.get('status')}")
        record = recommendations.get(feat, {"type": "numeric"})
        record["min"] = details["min"]
        record["max"] = details["max"]
        record["status"] = details.get("status", "normal-found")
        examples = record.get("examples", [])
        for val in details.get("examples", []):
            if val not in examples: examples.append(val)
        record["examples"] = examples[:3]
        if record.get("method") != "binary_multi":
            record["method"] = "binary_search"
        recommendations[feat] = record
        prob_val = details.get("best_prob")
        if prob_val is not None:
            best_prob = prob_val if best_prob is None else min(best_prob, prob_val)
    
    # 3) 범주형 후보 평가
    if categorical_targets:
        print(f"\n--- 3단계: 범주형 후보 평가 ---")
        for feat in categorical_targets:
            meta = input_metadata.get(feat)
            if not meta:
                print(f"  ✗ {feat}: 메타데이터 없음")
                continue
            choices = meta.get("choices", [])
            if not choices:
                print(f"  ✗ {feat}: 선택지 없음")
                continue
            result = evaluate_categorical_candidates(base_row, feat, choices, top_k=3)
            if not result:
                continue
            recommendations[feat] = {
                "type": "categorical",
                "values": result["values"],
                "status": result.get("status", "normal-found")
            }
            if result.get("best_prob") is not None:
                best_prob = result["best_prob"] if best_prob is None else min(best_prob, result["best_prob"])
    
    if best_prob is not None:
        recommendations["best_probability"] = float(best_prob)
    
    print(f"\n{'='*60}")
    print(f"📋 최종 추천 결과")
    print(f"{'='*60}")
    if recommendations:
        print(f"✅ 추천 항목: {len([k for k in recommendations.keys() if k != 'best_probability'])}개")
        for feat, rec in recommendations.items():
            if feat == "best_probability": continue
            if rec.get("type") == "numeric":
                print(f"  [수치형] {feat}: [{rec.get('min'):.4f}, {rec.get('max'):.4f}] ({rec.get('status')})")
            elif rec.get("type") == "categorical":
                print(f"  [범주형] {feat}: {rec.get('values')} ({rec.get('status')})")
        if "best_probability" in recommendations:
            print(f"\n예상 불량 확률(추천 적용): {recommendations['best_probability']:.4f}")
    else:
        print(f"✗ 추천 구간 없음")
    print(f"{'='*60}\n")
    return recommendations

# ===================== SHAP 예측 + 추천 연결 =====================

def predict_with_shap(row_dict):
    if not MODEL_LOADED:
        return None
    try:
        input_df = build_input_dataframe(row_dict)
        feature_matrix = prepare_feature_matrix(input_df)
        if feature_matrix is None:
            return None

        iteration = _booster_iteration(model)
        probs = model.predict(feature_matrix, num_iteration=iteration) if iteration else model.predict(feature_matrix)
        probability = float(probs[0])
        prediction = 1 if probability >= threshold else 0

        forced_fail = False
        tryshot = row_dict.get("tryshot_signal")
        if tryshot is not None and str(tryshot).upper() == "D":
            if probability < threshold:
                forced_fail = True
            prediction = 1

        contributions, shap_vector = compute_shap_contributions(feature_matrix)
        explanation = build_shap_explanation(contributions, shap_vector, row_dict)

        top_features = []
        if contributions:
            positive_items = [(f, c) for f, c in contributions.items() if c > 0]
            if positive_items:
                positive_items.sort(key=lambda x: abs(x[1]), reverse=True)
                items = positive_items[:5]
                print(f"\n✓ 양수 기여 변수 정렬: 상위 5개 선택")
            else:
                items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                print(f"\n✓ 전체 변수 정렬: 상위 5개 선택 (양수 기여 없음)")

            for feat, contrib in items:
                top_features.append({
                    "name": feat,
                    "label": COLUMN_NAMES_KR.get(feat, feat),
                    "value": row_dict.get(feat, np.nan),
                    "contribution": contrib
                })
        
        print(f"\n📊 상위 5개 변수 (SHAP 크기 순):")
        for rank, item in enumerate(top_features, 1):
            feat_type = "수치형" if item["name"] in ui_numeric_cols else "범주형"
            in_ranges = "O" if item["name"] in NUMERIC_FEATURE_RANGES else "X"
            print(f"  {rank}. [{feat_type}] {item['label']}: RANGES={in_ranges}, SHAP={item['contribution']:+.4f}")

        # 추천은 UI 스키마 기준, 예측은 모델 스키마 기준으로 처리됨
        recommendations = recommend_ranges(row_dict, [item["name"] for item in top_features])

        return {
            "probability": probability, "prediction": prediction, "forced_fail": forced_fail,
            "contributions": contributions, "shap_vector": shap_vector, "explanation": explanation,
            "top_features": top_features, "recommendations": recommendations, "input_row": row_dict
        }
    except Exception as e:
        print(f"❌ SHAP 예측 실패: {e}")
        import traceback; traceback.print_exc()
        return None

# ===================== UI =====================

tab_ui = ui.page_fluid(
    ui.div(
        ui.div(ui.output_ui("tab_log_stats"),
               style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),

        ui.div(
            ui.div(ui.HTML('<i class="fa-solid fa-table-list"></i> 누적 데이터 (전체)'),
                   style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #e0e0e0"),
            ui.output_ui("tab_log_table_all_wrapper"),
            style="background:#fff;border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),

        ui.div(
            ui.div(ui.HTML('<i class="fa-solid fa-exclamation-circle"></i> 누적 데이터 (불량)'),
                   style="font-size:18px;font-weight:700;color:#e74c3c;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #e74c3c"),
            ui.output_ui("tab_log_table_defect_wrapper"),
            style="background:#fff;border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),

        ui.div(
            ui.div(ui.HTML('<i class="fa-solid fa-chart-bar"></i> SHAP 변수 영향도 측정'),
                   style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #2A2D30"),
            ui.output_ui("shap_info_message"),
            ui.output_plot("shap_waterfall_plot", height="600px"),
            ui.output_ui("shap_analysis_details"),
            style="background:#fff;border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),

        style="max-width:1400px;margin:0 auto;padding:20px 0"),
)

def tab_server(input, output, session, streamer, shared_df, streaming_active):

    selected_row_data = reactive.Value(None)
    analysis_result = reactive.Value(None)

    @output
    @render.ui
    def tab_log_stats():
        try:
            df = shared_df.get()
            if df.empty:
                return ui.div("데이터 없음", style="color:#6c757d")

            predictions = predict_passorfail(df)

            temp_df = df.copy()
            temp_df['passorfail'] = predictions
            
            update_numeric_feature_ranges(temp_df)
            create_input_metadata(df)

            total_rows = len(df)
            memory_usage = f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            defect_count = int(predictions.sum())

            return ui.div(
                ui.div(ui.HTML(f'<i class="fa-solid fa-list-ol"></i> 총 데이터 행: {total_rows:,}'),
                       style="font-weight:600;font-size:16px;color:#2c3e50"),
                ui.div(ui.HTML(f'<i class="fa-solid fa-memory"></i> 메모리 사용량: {memory_usage}'),
                       style="font-weight:600;font-size:16px;color:#2c3e50;margin-top:10px"),
                ui.div(ui.HTML(f'<i class="fa-solid fa-exclamation-triangle"></i> 불량 건수: {defect_count:,}'),
                       style="font-weight:600;font-size:16px;color:#e74c3e;margin-top:10px"))
        except Exception as e:
            print(f"통계 오류: {e}")
            import traceback; traceback.print_exc()
            return ui.div(f"통계 계산 오류", style="color:#dc3545")

    @output
    @render.ui
    def tab_log_table_all_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...", style="text-align:center;padding:40px;color:#666"),
                         style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")

        return ui.div(ui.output_data_frame("tab_log_table_all"),
                     style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:600px")

    @output
    @render.data_frame
    def tab_log_table_all():
        try:
            df = shared_df.get()
            if df.empty:
                return render.DataGrid(pd.DataFrame(), height="600px", width="100%")

            result = df.copy()
            result['passorfail'] = predict_passorfail(result)
            result = result.drop(columns=DROP_COLUMNS, errors='ignore')

            return render.DataGrid(result, height="600px", width="100%", filters=False, row_selection_mode="none")
        except Exception as e:
            print(f"테이블 오류: {e}")
            return render.DataGrid(pd.DataFrame({"오류": [str(e)]}), height="600px", width="100%")

    @output
    @render.ui
    def tab_log_table_defect_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...", style="text-align:center;padding:40px;color:#666"),
                         style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")

        return ui.div(ui.output_data_frame("tab_log_table_defect"),
                     style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:600px")

    @output
    @render.data_frame
    def tab_log_table_defect():
        try:
            df = shared_df.get()
            if df.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["데이터 없음"]}), height="600px", width="100%")

            result = df.copy()
            result['passorfail'] = predict_passorfail(result)

            defect_only = result[result['passorfail'] == 1].copy()
            defect_only = defect_only.drop(columns=DROP_COLUMNS, errors='ignore')

            if defect_only.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["✅ 불량 없음"]}), height="600px", width="100%")

            return render.DataGrid(defect_only, height="600px", width="100%", filters=False, row_selection_mode="single")
        except Exception as e:
            print(f"불량 테이블 오류: {e}")
            return render.DataGrid(pd.DataFrame({"오류": [str(e)]}), height="600px", width="100%")

    @reactive.effect
    def _handle_defect_row_selection():
        try:
            selected = input.tab_log_table_defect_selected_rows()

            if not selected:
                selected_row_data.set(None)
                analysis_result.set(None)
                return

            df = shared_df.get()
            if df.empty:
                return

            result = df.copy()
            result['passorfail'] = predict_passorfail(result)

            print(f"\n{'='*60}")
            print(f"🔄 불량 행 선택 시 범위 재업데이트")
            print(f"{'='*60}")
            update_numeric_feature_ranges(result)

            defect_only = result[result['passorfail'] == 1].copy()
            if defect_only.empty:
                return

            idx = list(selected)[0]
            if idx >= len(defect_only):
                return

            row = defect_only.iloc[idx]
            selected_row_data.set(row)

            print(f"\n📌 선택된 불량 행 인덱스: {idx}")
            shap_result = predict_with_shap(row.to_dict())
            if shap_result:
                analysis_result.set(shap_result)
                print(f"✅ SHAP 분석 완료\n")
            else:
                analysis_result.set(None)
                print(f"❌ SHAP 분석 실패\n")
        except Exception as e:
            print(f"❌ 행 선택 오류: {e}")
            import traceback; traceback.print_exc()

    @output
    @render.ui
    def shap_info_message():
        result = analysis_result.get()

        if result is None:
            return ui.div("불량 행을 선택하면 SHAP 분석이 표시됩니다.",
                         style="text-align:center;padding:40px;color:#6c757d")

        prob = result['probability']
        pred_label = "불량" if result['prediction'] == 1 else "정상"
        pred_color = "#dc3545" if pred_label == "불량" else "#28a745"

        forced_msg = ''
        if result['forced_fail']:
            forced_msg = (
                '<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:12px;'
                'margin-top:10px;border-radius:4px;">⚠️ <strong>tryshot_signal 규칙 적용(실사용 판단)</strong>'
                '<br><span style="font-size:12px;color:#6c757d;">※ 추천 탐색 중에는 강제불량을 일시 무시하여 '
                '정상/최소확률 구간을 탐색합니다.</span></div>'
            )

        return ui.HTML(
            f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-bottom:20px;">'
            f'<div style="font-size:24px;font-weight:700;color:{pred_color};margin-bottom:15px;">{pred_label}</div>'
            f'<div style="font-size:15px;margin-bottom:8px;">불량 확률: '
            f'<strong style="font-size:18px;color:#dc3545;">{prob:.4f}</strong> '
            f'(임계값: {threshold:.4f})</div>{forced_msg}</div>'
        )

    @output
    @render.plot
    def shap_waterfall_plot():
        result = analysis_result.get()

        fig, ax = plt.subplots(figsize=(10, 6))

        if result is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "불량 행을 선택하세요", ha="center", va="center", fontsize=14, color="#6c757d")
            plt.tight_layout()
            return fig

        explanation = result.get('explanation')
        if explanation is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "SHAP 생성 실패", ha="center", va="center", fontsize=14, color="#dc3545")
            plt.tight_layout()
            return fig

        try:
            plt.close('all')
            setup_korean_font()
            shap.plots.waterfall(explanation, max_display=20, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Plot 오류: {e}")
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")
            ax.text(0.5, 0.5, "Plot 생성 실패", ha="center", va="center", fontsize=12, color="#dc3545")
            plt.tight_layout()
            return fig

    @output
    @render.ui
    def shap_analysis_details():
        result = analysis_result.get()
        if result is None:
            return ui.div()

        top_features = result.get('top_features', [])
        recommendations = result.get('recommendations', {})
        if not top_features:
            return ui.div()

        items = []
        for rank, item in enumerate(top_features, 1):
            feat_name = item["name"]
            val = format_value(item.get("value"))
            contrib = item.get("contribution", 0.0)
            direction = "위험 증가" if contrib > 0 else "위험 감소"
            color = "#dc3545" if contrib > 0 else "#28a745"

            # ⬇️ 불필요 라벨/배지 제거 (수치형/범주형, 다변수 최적화, 임계값 미달)
            item_html = (
                f'<li style="margin-bottom:16px;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="background:#2A2D30;color:white;border-radius:50%;width:24px;height:24px;'
                f'display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;">{rank}</span>'
                f'<strong>{item["label"]}</strong>'
                f'</div>'
                f'<div style="margin-left:32px;margin-top:4px;">'
                f'<span style="color:#6c757d;font-size:13px;">현재 값: {val}</span><br>'
            )

            # 추천 문구 (method/배지 문구 완전 제거, 2가지 케이스만)
            rec = recommendations.get(feat_name, {})
            if rec:
                if rec.get("type") == "numeric":
                    min_v = format_value(rec.get("min"))
                    max_v = format_value(rec.get("max"))
                    if rec.get("status") == "no-normal-but-best":
                        item_html += (
                            f'<span style="color:#856404;font-weight:600;">'
                            f'• 정상 전환 불가, 확률 최소화 후보: {min_v} ~ {max_v}'
                            f'</span><br>'
                        )
                    else:
                        item_html += (
                            f'<span style="color:#28a745;font-weight:600;">'
                            f'✓ 정상 전환 추천: {min_v} ~ {max_v}'
                            f'</span><br>'
                        )
                elif rec.get("type") == "categorical":
                    values = ", ".join(rec.get("values", []))
                    if rec.get("status") == "no-normal-but-best":
                        item_html += (
                            f'<span style="color:#856404;font-weight:600;">'
                            f'• 정상 전환 불가, 확률 최소화 후보: {values}'
                            f'</span><br>'
                        )
                    else:
                        item_html += (
                            f'<span style="color:#28a745;font-weight:600;">'
                            f'✓ 정상 전환 추천 값: {values}'
                            f'</span><br>'
                        )

            # SHAP 기여도만 표시
            item_html += (
                f'<span style="font-size:13px;">SHAP 기여도: </span>'
                f'<span style="color:{color};font-weight:600;font-size:14px;">{contrib:+.4f}</span> '
                f'<span style="color:#6c757d;font-size:13px;">({direction})</span>'
                f'</div>'
                f'</li>'
            )

            items.append(item_html)
        
        best_prob = recommendations.get("best_probability")
        prob_html = ""
        if best_prob is not None:
            prob_html = (
                f'<div style="background:#d4edda;border-left:4px solid #28a745;padding:12px;margin-top:16px;border-radius:4px;">'
                f'<strong>📈 추천 적용 시 예상 불량 확률(최소): {best_prob:.4f}</strong> '
                f'<span style="color:#6c757d;font-size:12px;">(현재 임계값: {threshold:.4f})</span>'
                f'</div>'
            )

        html = (
            f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-top:20px;">'
            f'<div style="font-weight:600;font-size:16px;margin-bottom:15px;border-bottom:2px solid #dee2e6;'
            f'padding-bottom:8px;">📊 불량 영향 상위 변수 및 정상/최소확률 전환 권고 (SHAP 크기 순)</div>'
            f'<ul style="padding-left:20px;margin:0;">{"".join(items)}</ul>'
            f'{prob_html}'
            f'</div>'
        )
        return ui.HTML(html)
