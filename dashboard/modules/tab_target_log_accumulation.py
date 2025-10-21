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

# ========== 한글 폰트 설정 (마이너스 지원) ==========
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
BASE_DIR = Path(__file__).resolve().parents[1]
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

# 모델 로드
try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    ordinal_encoder = artifact.get("ordinal_encoder")
    onehot_encoder = artifact.get("onehot_encoder")
    threshold = float(artifact.get("operating_threshold", 0.5))

    numeric_cols = list(scaler.feature_names_in_) if scaler and hasattr(scaler, 'feature_names_in_') else []
    categorical_cols = list(ordinal_encoder.feature_names_in_) if ordinal_encoder and hasattr(ordinal_encoder, 'feature_names_in_') else []
    required_cols = numeric_cols + categorical_cols

    explainer = shap.TreeExplainer(model)
    numeric_index_map = {feat: idx for idx, feat in enumerate(numeric_cols)}

    ohe_feature_slices = {}
    start_idx = len(numeric_cols)
    if categorical_cols and onehot_encoder is not None:
        for feat, ohe_cats in zip(categorical_cols, onehot_encoder.categories_):
            length = len(ohe_cats)
            ohe_feature_slices[feat] = (start_idx, start_idx + length)
            start_idx += length

    NUMERIC_FEATURE_RANGES = {}
    input_metadata = {}
    MODEL_LOADED = True

    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    MODEL_LOADED = False
    model = None
    explainer = None
    required_cols = []
    numeric_cols = []
    categorical_cols = []
    NUMERIC_FEATURE_RANGES = {}
    input_metadata = {}

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

def predict_passorfail(df):
    if not MODEL_LOADED or df.empty:
        return np.zeros(len(df), dtype=int)

    try:
        X = df.drop(columns=DROP_COLUMNS + ['passorfail'], errors='ignore').copy()

        for col in required_cols:
            if col not in X.columns:
                X[col] = 0.0 if col in numeric_cols else 'UNKNOWN'

        X = X[required_cols].copy()

        if numeric_cols and scaler is not None:
            X_num = X[numeric_cols].fillna(0.0)
            X_num_scaled = scaler.transform(X_num)
        else:
            X_num_scaled = np.empty((len(X), 0))

        if categorical_cols and ordinal_encoder is not None and onehot_encoder is not None:
            X_cat = X[categorical_cols].fillna('UNKNOWN')
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

        if 'tryshot_signal' in df.columns:
            predictions[df['tryshot_signal'].to_numpy() == 'D'] = 1

        return predictions
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        return np.zeros(len(df), dtype=int)

# ========== 핵심 수정: 3단계 폴백 시스템 ==========
def update_numeric_feature_ranges(df):
    """3단계 폴백으로 안전하게 범위 업데이트"""
    global NUMERIC_FEATURE_RANGES
    
    if df.empty:
        print("⚠️ 빈 데이터프레임 - 범위 업데이트 건너뜀")
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"🔧 NUMERIC_FEATURE_RANGES 업데이트 시작")
        print(f"{'='*60}")
        print(f"전체 데이터: {len(df)}행")
        
        # ===== 1단계: 정상 데이터 우선 (올바른 필터링) =====
        if "passorfail" in df.columns:
            # 수정: df.get() → df[컬럼] 직접 접근
            pass_df = df[df["passorfail"] == 0].copy()
            print(f"✓ 1단계: passorfail 컬럼 존재")
            print(f"  - 정상 데이터(0): {len(pass_df)}행")
            print(f"  - 불량 데이터(1): {len(df) - len(pass_df)}행")
        else:
            pass_df = pd.DataFrame()
            print(f"⚠️ 1단계: passorfail 컬럼 없음")
        
        # ===== 2단계: 전체 데이터 폴백 =====
        if pass_df.empty:
            pass_df = df.copy()
            print(f"⚠️ 2단계: 정상 데이터 없음 → 전체 데이터 사용 ({len(pass_df)}행)")
        
        NUMERIC_FEATURE_RANGES = {}
        success_cols = []
        failed_cols = []
        
        for col in numeric_cols:
            if col not in pass_df.columns:
                failed_cols.append(f"{col} (컬럼 없음)")
                continue
            
            # 숫자 변환 시도
            series = pd.to_numeric(pass_df[col], errors='coerce').dropna()
            
            # 실패 시 전체 데이터에서 재시도
            if series.empty and col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if not series.empty:
                min_val, max_val = float(series.min()), float(series.max())
                
                # 동일값 처리
                if min_val == max_val:
                    if min_val != 0:
                        min_val = min_val - abs(min_val * 0.1)
                        max_val = max_val + abs(max_val * 0.1)
                    else:
                        min_val = -1.0
                        max_val = 1.0
                
                NUMERIC_FEATURE_RANGES[col] = (min_val, max_val)
                success_cols.append(col)
            else:
                failed_cols.append(f"{col} (변환 실패)")
        
        # ===== 3단계: 기본값 폴백 (최후의 수단) =====
        if not NUMERIC_FEATURE_RANGES:
            print(f"❌ 3단계: 모든 변수 변환 실패 - 기본 범위 사용")
            for col in numeric_cols:
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
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 최소한의 기본값 설정
        if not NUMERIC_FEATURE_RANGES:
            print(f"⚠️ 에러 복구: 기본 범위 설정")
            for col in numeric_cols:
                NUMERIC_FEATURE_RANGES[col] = (0.0, 100.0)
            print(f"✓ 기본 범위: {len(NUMERIC_FEATURE_RANGES)}개")

def create_input_metadata(df):
    """입력 메타데이터 생성"""
    global input_metadata
    
    metadata = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        values = sorted([str(v) for v in df[col].dropna().unique()])
        if values:
            metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            continue
        vmin, vmax = float(s.min()), float(s.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": float(s.median())}
    
    input_metadata = metadata

def build_input_dataframe(row_dict):
    data = {col: row_dict.get(col) for col in required_cols}
    input_df = pd.DataFrame([data], columns=required_cols)
    if numeric_cols:
        input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        input_df[categorical_cols] = input_df[categorical_cols].fillna("UNKNOWN").astype(str)
    return input_df

def prepare_feature_matrix(input_df):
    if not MODEL_LOADED:
        return None

    try:
        arrays = []
        if numeric_cols and scaler is not None:
            arrays.append(scaler.transform(input_df[numeric_cols].astype(float)))
        if categorical_cols and ordinal_encoder is not None and onehot_encoder is not None:
            cat_ord = ordinal_encoder.transform(input_df[categorical_cols]).astype(int)
            cat_ohe = onehot_encoder.transform(cat_ord)
            if hasattr(cat_ohe, "toarray"):
                cat_ohe = cat_ohe.toarray()
            arrays.append(cat_ohe)

        return np.hstack(arrays).astype(np.float32) if arrays else np.zeros((len(input_df), 0), dtype=np.float32)
    except:
        return None

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
    except:
        return None, None

def _extract_feature_values(row_dict):
    values = []
    for col in required_cols:
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

        shap_values = np.array([float(contributions.get(col, 0.0)) for col in required_cols], dtype=float)
        feature_values = np.array(_extract_feature_values(input_row), dtype=object)
        feature_names = [COLUMN_NAMES_KR.get(col, col) for col in required_cols]

        return Explanation(values=shap_values, base_values=expected_value, data=feature_values, feature_names=feature_names)
    except:
        return None

def evaluate_prediction(row_dict):
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
        prediction = 1 if probability >= threshold else 0

        tryshot = row_dict.get("tryshot_signal")
        if tryshot is not None and str(tryshot).upper() == "D":
            prediction = 1

        return prediction, probability
    except:
        return 0, 0.0

def find_normal_range_binary_fixed(base_row, feature, bounds, max_iter=15, n_check=5):
    if not bounds:
        return None

    f_min, f_max = bounds
    if pd.isna(f_min) or pd.isna(f_max) or f_min >= f_max:
        return None

    low, high = float(f_min), float(f_max)
    best_details = None

    for _ in range(max_iter):
        samples = np.linspace(low, high, n_check)
        normal_samples = []

        for val in samples:
            trial = base_row.copy()
            trial[feature] = float(val)
            pred, prob = evaluate_prediction(trial)
            if pred == 0:
                normal_samples.append((float(val), float(prob)))

        if not normal_samples:
            break

        normal_samples.sort(key=lambda x: x[1])
        low = min(v for v, _ in normal_samples)
        high = max(v for v, _ in normal_samples)
        top_val, top_prob = normal_samples[0]

        if best_details is None or top_prob < best_details[3]:
            best_details = (low, high, [top_val, low, high][:3], top_prob)

        if (high - low) <= 0.01:
            break

    if best_details:
        return {"min": best_details[0], "max": best_details[1], "examples": best_details[2], "best_prob": best_details[3]}
    return None

# ========== 긴급 조치: 범위가 없을 때 현장 생성 ==========
def recommend_ranges(base_row, focus_features):
    """추천 구간 계산 (NUMERIC_FEATURE_RANGES 없어도 작동)"""
    if not focus_features:
        return {}
    
    # ===== 긴급 조치: 범위가 없으면 현장에서 생성 =====
    if not NUMERIC_FEATURE_RANGES:
        print(f"\n⚠️ NUMERIC_FEATURE_RANGES 비어있음 - 임시 범위 생성")
        
        temp_ranges = {}
        for feat in focus_features:
            if feat in numeric_cols:
                current_val = base_row.get(feat)
                if current_val is not None and not pd.isna(current_val):
                    try:
                        current_val = float(current_val)
                        # 현재 값 기준 ±30% 범위
                        if current_val != 0:
                            temp_ranges[feat] = (
                                current_val * 0.7,
                                current_val * 1.3
                            )
                        else:
                            temp_ranges[feat] = (-10.0, 10.0)
                        print(f"  {feat}: [{temp_ranges[feat][0]:.4f}, {temp_ranges[feat][1]:.4f}] (현재값 기준)")
                    except (ValueError, TypeError):
                        continue
        
        if temp_ranges:
            recommendations = {}
            for feat, bounds in temp_ranges.items():
                details = find_normal_range_binary_fixed(base_row, feat, bounds)
                if details:
                    recommendations[feat] = {
                        "type": "numeric",
                        "min": details["min"],
                        "max": details["max"],
                        "examples": details["examples"],
                        "best_prob": details["best_prob"]
                    }
                    print(f"  ✓ {feat}: 추천 구간 생성 성공")
            
            return recommendations
        else:
            print(f"  ✗ 임시 범위 생성 실패")
            return {}
    # ================================================
    
    recommendations = {}
    
    for feat in focus_features:
        if feat not in numeric_cols or feat not in NUMERIC_FEATURE_RANGES:
            continue
        
        bounds = NUMERIC_FEATURE_RANGES[feat]
        details = find_normal_range_binary_fixed(base_row, feat, bounds)
        
        if details:
            recommendations[feat] = {
                "type": "numeric",
                "min": details["min"],
                "max": details["max"],
                "examples": details["examples"],
                "best_prob": details["best_prob"]
            }
    
    return recommendations

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
            items = (positive_items[:5] if positive_items else sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

            for feat, contrib in items:
                top_features.append({
                    "name": feat,
                    "label": COLUMN_NAMES_KR.get(feat, feat),
                    "value": row_dict.get(feat, np.nan),
                    "contribution": contrib
                })
        
        # ===== 디버깅: 상위 변수 타입 확인 =====
        print(f"\n📊 상위 5개 변수:")
        for item in top_features:
            feat_type = "수치형" if item["name"] in numeric_cols else "범주형"
            in_ranges = "O" if item["name"] in NUMERIC_FEATURE_RANGES else "X"
            print(f"  [{feat_type}] {item['label']}: RANGES={in_ranges}")
        # ======================================

        recommendations = recommend_ranges(row_dict, [item["name"] for item in top_features])
        
        # ===== 디버깅: 추천 구간 결과 =====
        print(f"\n🔧 추천 구간 계산 결과:")
        if recommendations:
            for feat, rec in recommendations.items():
                print(f"  {feat}: [{rec.get('min'):.4f}, {rec.get('max'):.4f}]")
        else:
            print(f"  ✗ 추천 구간 없음")
        # ===================================

        return {
            "probability": probability, "prediction": prediction, "forced_fail": forced_fail,
            "contributions": contributions, "shap_vector": shap_vector, "explanation": explanation,
            "top_features": top_features, "recommendations": recommendations, "input_row": row_dict
        }
    except Exception as e:
        print(f"❌ SHAP 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            
            # ===== 중요: 통계 표시 시점에 범위 업데이트 =====
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
                       style="font-weight:600;font-size:16px;color:#e74c3c;margin-top:10px"))
        except Exception as e:
            print(f"통계 오류: {e}")
            import traceback
            traceback.print_exc()
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

            # ===== 중요: 선택 시점에도 범위 강제 업데이트 =====
            print(f"\n{'='*60}")
            print(f"🔄 불량 행 선택 시 범위 재업데이트")
            print(f"{'='*60}")
            update_numeric_feature_ranges(result)
            # ==============================================

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
            import traceback
            traceback.print_exc()

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
            forced_msg = '<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:12px;margin-top:10px;border-radius:4px;">⚠️ <strong>tryshot_signal 규칙 적용</strong></div>'

        return ui.HTML(
            f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-bottom:20px;">'
            f'<div style="font-size:24px;font-weight:700;color:{pred_color};margin-bottom:15px;">{pred_label}</div>'
            f'<div style="font-size:15px;margin-bottom:8px;">불량 확률: <strong style="font-size:18px;color:#dc3545;">{prob:.4f}</strong> (임계값: {threshold:.4f})</div>'
            f'{forced_msg}</div>')

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

        # 불량 영향 상위 변수 + 정상 전환 추천 구간 통합
        items = []
        for item in top_features:
            feat_name = item["name"]
            val = format_value(item.get("value"))
            contrib = item.get("contribution", 0.0)
            direction = "위험 증가" if contrib > 0 else "위험 감소"
            color = "#dc3545" if contrib > 0 else "#28a745"

            item_html = (
                f'<li style="margin-bottom:12px;">'
                f'<strong>{item["label"]}</strong> '
                f'<span style="color:#6c757d;">(현재: {val}</span>'
            )

            # 정상 전환 추천 구간 추가 (수치형만)
            if feat_name in recommendations:
                rec = recommendations[feat_name]
                if rec.get("type") == "numeric":
                    min_v = format_value(rec.get("min"))
                    max_v = format_value(rec.get("max"))
                    item_html += f'<span style="color:#6c757d;">, 정상 전환 추천 구간: </span>'
                    item_html += f'<span style="color:#28a745;font-weight:600;">{min_v} ~ {max_v}</span>'

            item_html += f'<span style="color:#6c757d;">)</span><br>'
            item_html += (
                f'<span style="font-size:13px;">SHAP: </span>'
                f'<span style="color:{color};font-weight:600;font-size:14px;">{contrib:+.4f}</span> '
                f'<span style="color:#6c757d;font-size:13px;">({direction})</span>'
                f'</li>'
            )

            items.append(item_html)

        html = (
            f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-top:20px;">'
            f'<div style="font-weight:600;font-size:16px;margin-bottom:15px;border-bottom:2px solid #dee2e6;padding-bottom:8px;">'
            f'📊 불량 영향 상위 변수 및 정상 전환 추천 구간</div>'
            f'<ul style="padding-left:20px;margin:0;">{"".join(items)}</ul>'
            f'</div>'
        )

        return ui.HTML(html)