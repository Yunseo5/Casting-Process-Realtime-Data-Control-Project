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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import base64

# ========== 로깅 설정 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 상수 정의 ==========
@dataclass
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    MODEL_PATH: Path = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"
    PROCESS_IMAGE_PATH: Path = BASE_DIR / "data" / "illustration" / "Casting_process_illustration.png"
    DROP_COLUMNS: List[str] = None
    DEFAULT_THRESHOLD: float = 0.7553
    PERCENTILE_LOW: float = 0.01
    PERCENTILE_HIGH: float = 0.99
    RANGE_BUFFER: float = 0.05
    MAX_ITER_BINARY: int = 15
    MAX_ITER_MULTI: int = 10
    TOLERANCE_RATIO: float = 0.01
    TOP_K_CATEGORICAL: int = 3
    TOP_FEATURES_COUNT: int = 5
    
    def __post_init__(self):
        self.DROP_COLUMNS = ['line', 'name', 'mold_name', 'date', 'time', 'Unnamed: 0', 'id']

config = Config()

# ========== 공정 프로세스 매핑 ==========
PROCESS_MAPPING = {
    "용탕": {
        "variables": ["molten_temp", "EMS_operation_time"],
        "position": {"top": "90%", "left": "50%"}
    },
    "사출": {
        "variables": ["cast_pressure", "low_section_speed", "high_section_speed", 
                     "sleeve_temperature", "biscuit_thickness"],
        "position": {"top": "50%", "left": "40%"}
    },
    "금형": {
        "variables": ["upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1", 
                     "lower_mold_temp2", "Coolant_temperature", "physical_strength"],
        "position": {"top": "40%", "left": "70%"}
    },
    "운영 메트릭": {
        "variables": ["production_cycletime", "count"],
        "position": {"top": "10%", "left": "90%"}
    }
}

SHAP_THRESHOLD_CRITICAL = 0.15
SHAP_THRESHOLD_WARNING = 0.05

COLUMN_NAMES_KR = {
    "registration_time": "등록 일시", "count": "생산 순번", "working": "가동 여부",
    "emergency_stop": "비상 정지", "facility_operation_cycleTime": "설비 운영 사이클타임",
    "production_cycletime": "제품 생산 사이클타임", "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도", "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스킷 두께", "upper_mold_temp1": "상부 금형 온도1",
    "upper_mold_temp2": "상부 금형 온도2", "upper_mold_temp3": "상부 금형 온도3",
    "lower_mold_temp1": "하부 금형 온도1", "lower_mold_temp2": "하부 금형 온도2",
    "lower_mold_temp3": "하부 금형 온도3", "sleeve_temperature": "슬리브 온도",
    "physical_strength": "형체력", "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "전자교반 가동시간", "mold_code": "금형 코드",
    "tryshot_signal": "트라이샷 신호", "molten_temp": "용탕 온도",
    "molten_volume": "용탕 부피", "heating_furnace": "가열로",
    "passorfail": "불량 여부", "uniformity": "균일도",
    "mold_temp_udiff": "금형 온도차(상/하)", "P_diff": "압력 차이",
    "Cycle_diff": "사이클 시간 차이"
}

# ========== 한글 폰트 설정 ==========
def setup_korean_font() -> None:
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    korean_fonts = ['Noto Sans KR', 'Noto Sans CJK KR', 'NanumGothic', 'AppleGothic', 'Malgun Gothic']
    chosen = next((f for f in korean_fonts if f in available_fonts), None)
    plt.rcParams['font.family'] = [chosen, 'DejaVu Sans'] if chosen else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True

setup_korean_font()

# ========== 모델 클래스 ==========
class DefectPredictionModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.loaded = False
        self.model = None
        self.scaler = None
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.threshold = config.DEFAULT_THRESHOLD
        self.explainer = None
        self.model_numeric_cols: List[str] = []
        self.model_categorical_cols: List[str] = []
        self.model_required_cols: List[str] = []
        self.ui_numeric_cols: List[str] = []
        self.ui_categorical_cols: List[str] = []
        self.numeric_feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.input_metadata: Dict[str, Any] = {}
        self.numeric_index_map: Dict[str, int] = {}
        self.ohe_feature_slices: Dict[str, Tuple[int, int]] = {}
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.scaler = artifact.get("scaler")
            self.ordinal_encoder = artifact.get("ordinal_encoder")
            self.onehot_encoder = artifact.get("onehot_encoder")
            
            threshold = artifact.get("operating_threshold")
            if threshold is None or abs(float(threshold) - 0.5) < 0.001:
                self.threshold = config.DEFAULT_THRESHOLD
            else:
                self.threshold = float(threshold)
            
            self._initialize_schema()
            self._initialize_shap()
            self.loaded = True
            logger.info(f"모델 로드 성공 (임계값: {self.threshold})")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.loaded = False
    
    def _initialize_schema(self) -> None:
        if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
            self.model_numeric_cols = list(self.scaler.feature_names_in_)
        if self.ordinal_encoder and hasattr(self.ordinal_encoder, 'feature_names_in_'):
            self.model_categorical_cols = list(self.ordinal_encoder.feature_names_in_)
        self.model_required_cols = self.model_numeric_cols + self.model_categorical_cols
        self.ui_numeric_cols = self.model_numeric_cols.copy()
        self.ui_categorical_cols = self.model_categorical_cols.copy()
    
    def _initialize_shap(self) -> None:
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.numeric_index_map = {feat: idx for idx, feat in enumerate(self.model_numeric_cols)}
            start_idx = len(self.model_numeric_cols)
            if self.model_categorical_cols and self.onehot_encoder:
                for feat, ohe_cats in zip(self.model_categorical_cols, self.onehot_encoder.categories_):
                    length = len(ohe_cats)
                    self.ohe_feature_slices[feat] = (start_idx, start_idx + length)
                    start_idx += length
        except Exception as e:
            logger.error(f"SHAP 초기화 실패: {e}")
    
    def get_booster_iteration(self) -> Optional[int]:
        iteration = getattr(self.model, "best_iteration", None)
        if not iteration:
            current_iteration = getattr(self.model, "current_iteration", None)
            if callable(current_iteration):
                iteration = current_iteration()
        return int(iteration) if iteration else None

model_manager = DefectPredictionModel(config.MODEL_PATH)

# ========== 유틸리티 함수 ==========
def format_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)

def safe_numeric_conversion(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def get_process_image_base64() -> str:
    try:
        if config.PROCESS_IMAGE_PATH.exists():
            with open(config.PROCESS_IMAGE_PATH, 'rb') as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
    return ""

# ========== 데이터 전처리 ==========
class DataPreprocessor:
    @staticmethod
    def build_input_dataframe(row_dict: Dict[str, Any]) -> pd.DataFrame:
        data = {col: row_dict.get(col) for col in model_manager.model_required_cols}
        input_df = pd.DataFrame([data], columns=model_manager.model_required_cols)
        if model_manager.model_numeric_cols:
            input_df[model_manager.model_numeric_cols] = input_df[model_manager.model_numeric_cols].apply(pd.to_numeric, errors="coerce")
        if model_manager.model_categorical_cols:
            input_df[model_manager.model_categorical_cols] = input_df[model_manager.model_categorical_cols].fillna("UNKNOWN").astype(str)
        return input_df
    
    @staticmethod
    def prepare_feature_matrix(input_df: pd.DataFrame) -> Optional[np.ndarray]:
        if not model_manager.loaded:
            return None
        try:
            arrays = []
            if model_manager.model_numeric_cols and model_manager.scaler:
                arrays.append(model_manager.scaler.transform(input_df[model_manager.model_numeric_cols].astype(float)))
            if model_manager.model_categorical_cols and model_manager.ordinal_encoder and model_manager.onehot_encoder:
                cat_ord = model_manager.ordinal_encoder.transform(input_df[model_manager.model_categorical_cols]).astype(int)
                cat_ohe = model_manager.onehot_encoder.transform(cat_ord)
                if hasattr(cat_ohe, "toarray"):
                    cat_ohe = cat_ohe.toarray()
                arrays.append(cat_ohe)
            return np.hstack(arrays).astype(np.float32) if arrays else np.zeros((len(input_df), 0), dtype=np.float32)
        except Exception as e:
            logger.error(f"특성 행렬 준비 실패: {e}")
            return None

preprocessor = DataPreprocessor()

# ========== 예측 클래스 ==========
class Predictor:
    @staticmethod
    def predict_batch(df: pd.DataFrame) -> np.ndarray:
        if not model_manager.loaded or df.empty:
            return np.zeros(len(df), dtype=int)
        try:
            X = df.drop(columns=config.DROP_COLUMNS + ['passorfail'], errors='ignore').copy()
            for col in model_manager.model_required_cols:
                if col not in X.columns:
                    X[col] = 0.0 if col in model_manager.model_numeric_cols else 'UNKNOWN'
            X = X[model_manager.model_required_cols].copy()
            arrays = []
            if model_manager.model_numeric_cols and model_manager.scaler:
                arrays.append(model_manager.scaler.transform(X[model_manager.model_numeric_cols].fillna(0.0)))
            if model_manager.model_categorical_cols and model_manager.ordinal_encoder and model_manager.onehot_encoder:
                X_cat_ord = model_manager.ordinal_encoder.transform(X[model_manager.model_categorical_cols].fillna('UNKNOWN')).astype(int)
                arrays.append(model_manager.onehot_encoder.transform(X_cat_ord))
            if not arrays:
                return np.zeros(len(df), dtype=int)
            X_final = np.hstack(arrays)
            iteration = model_manager.get_booster_iteration()
            probs = model_manager.model.predict(X_final, num_iteration=iteration) if iteration else model_manager.model.predict(X_final)
            predictions = (probs >= model_manager.threshold).astype(int)
            if 'tryshot_signal' in df.columns:
                predictions[df['tryshot_signal'].to_numpy() == 'D'] = 1
            return predictions
        except Exception as e:
            logger.error(f"배치 예측 실패: {e}")
            return np.zeros(len(df), dtype=int)
    
    @staticmethod
    def predict_single(row_dict: Dict[str, Any], ignore_tryshot: bool = False) -> Tuple[int, float]:
        if not model_manager.loaded:
            return 0, 0.0
        try:
            input_df = preprocessor.build_input_dataframe(row_dict)
            feature_matrix = preprocessor.prepare_feature_matrix(input_df)
            if feature_matrix is None:
                return 0, 0.0
            iteration = model_manager.get_booster_iteration()
            probs = model_manager.model.predict(feature_matrix, num_iteration=iteration) if iteration else model_manager.model.predict(feature_matrix)
            probability = float(probs[0])
            prediction = 1 if probability >= model_manager.threshold else 0
            if not ignore_tryshot and str(row_dict.get("tryshot_signal", "")).upper() == "D":
                prediction = 1
            return prediction, probability
        except Exception as e:
            logger.error(f"단일 예측 실패: {e}")
            return 0, 0.0

predictor = Predictor()

# ========== 범위 업데이트 ==========
class RangeUpdater:
    @staticmethod
    def update_ranges(df: pd.DataFrame) -> None:
        if df.empty:
            return
        try:
            pass_df = df[df["passorfail"] == 0] if "passorfail" in df.columns else df
            if pass_df.empty:
                pass_df = df
            model_manager.numeric_feature_ranges = {}
            for col in model_manager.ui_numeric_cols:
                if col not in pass_df.columns:
                    continue
                series = safe_numeric_conversion(pass_df[col]).dropna()
                if series.empty and col in df.columns:
                    series = safe_numeric_conversion(df[col]).dropna()
                if series.empty:
                    continue
                q_low, q_high = series.quantile(config.PERCENTILE_LOW), series.quantile(config.PERCENTILE_HIGH)
                span = q_high - q_low
                if not np.isfinite(span) or span <= 0:
                    min_val = q_low * 0.9 if q_low != 0 else -1.0
                    max_val = q_low * 1.1 if q_low != 0 else 1.0
                else:
                    min_val, max_val = float(q_low - config.RANGE_BUFFER * span), float(q_high + config.RANGE_BUFFER * span)
                model_manager.numeric_feature_ranges[col] = (min_val, max_val)
            if not model_manager.numeric_feature_ranges:
                for col in model_manager.ui_numeric_cols:
                    model_manager.numeric_feature_ranges[col] = (0.0, 100.0)
        except Exception as e:
            logger.error(f"범위 업데이트 실패: {e}")

range_updater = RangeUpdater()

# ========== 메타데이터 생성 ==========
class MetadataCreator:
    @staticmethod
    def create_metadata(df: pd.DataFrame) -> None:
        metadata = {}
        heuristic_categorical = [col for col in df.columns if str(col).lower() in ["mold_code", "mold", "code", "model_code"] or (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique(dropna=True) <= 20)]
        cat_set = set(model_manager.ui_categorical_cols) | set(heuristic_categorical)
        num_set = set(model_manager.ui_numeric_cols) - set(heuristic_categorical)
        for col in cat_set:
            if col in df.columns:
                values = sorted([str(v) for v in df[col].astype(str).dropna().unique()])
                if values:
                    metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}
        for col in num_set:
            if col in df.columns:
                s = safe_numeric_conversion(df[col]).dropna()
                if len(s) > 0:
                    vmin, vmax = float(s.quantile(config.PERCENTILE_LOW)), float(s.quantile(config.PERCENTILE_HIGH))
                    if vmin == vmax:
                        vmin, vmax = vmin - 1.0, vmax + 1.0
                    metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": float(s.median())}
        model_manager.ui_categorical_cols = list(cat_set)
        model_manager.ui_numeric_cols = list(num_set)
        model_manager.input_metadata = metadata

metadata_creator = MetadataCreator()

# ========== SHAP 분석 ==========
class SHAPAnalyzer:
    @staticmethod
    def compute_contributions(feature_matrix: np.ndarray) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        if not model_manager.loaded or model_manager.explainer is None or feature_matrix is None:
            return None, None
        try:
            shap_values = model_manager.explainer.shap_values(feature_matrix)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_vector = shap_values[0]
            contributions = {}
            for feat, idx in model_manager.numeric_index_map.items():
                contributions[feat] = float(shap_vector[idx])
            for feat, (start, end) in model_manager.ohe_feature_slices.items():
                contributions[feat] = float(np.sum(shap_vector[start:end])) if end > start else 0.0
            return contributions, shap_vector
        except Exception as e:
            logger.error(f"SHAP 기여도 계산 실패: {e}")
            return None, None
    
    @staticmethod
    def build_explanation(contributions: Dict, shap_vector: np.ndarray, input_row: Dict) -> Optional[Any]:
        if contributions is None or shap_vector is None:
            return None
        try:
            from shap import Explanation
            expected_value = float(model_manager.explainer.expected_value[1] if isinstance(model_manager.explainer.expected_value, (list, np.ndarray)) else model_manager.explainer.expected_value)
            shap_values = np.array([float(contributions.get(col, 0.0)) for col in model_manager.model_required_cols], dtype=float)
            feature_values = np.array([np.nan if pd.isna(val := input_row.get(col, np.nan)) else (val[0] if isinstance(val, (list, tuple)) and len(val) > 0 else (str(val) if isinstance(val, (pd.Timestamp, pd.Timedelta)) else val)) for col in model_manager.model_required_cols], dtype=object)
            feature_names = [COLUMN_NAMES_KR.get(col, col) for col in model_manager.model_required_cols]
            return Explanation(values=shap_values, base_values=expected_value, data=feature_values, feature_names=feature_names)
        except Exception as e:
            logger.error(f"SHAP explanation 생성 실패: {e}")
            return None

shap_analyzer = SHAPAnalyzer()

# ========== 추천 시스템 ==========
class RecommendationEngine:
    @staticmethod
    def find_normal_range_single(base_row: Dict, feature: str, bounds: Tuple[float, float]) -> Optional[Dict]:
        if not bounds or pd.isna(bounds[0]) or pd.isna(bounds[1]) or bounds[0] >= bounds[1]:
            return None
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        tol = max((high - low) * config.TOLERANCE_RATIO, 1e-3)
        best_details = None
        best_overall = None
        for _ in range(config.MAX_ITER_BINARY):
            samples = np.linspace(low, high, 5)
            normal_samples = []
            for val in samples:
                trial = base_row.copy()
                trial[feature] = float(val)
                pred, prob = predictor.predict_single(trial, ignore_tryshot=True)
                if best_overall is None or prob < best_overall[1]:
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
            if low not in examples:
                examples.append(low)
            if high not in examples:
                examples.append(high)
            if best_details is None or top_prob < best_details[3]:
                best_details = (low, high, examples[:3], top_prob)
            if (high - low) <= tol:
                break
        if best_details is None:
            if best_overall:
                return {"min": float(bounds[0]), "max": float(bounds[1]), "examples": [float(best_overall[0])], "best_prob": float(best_overall[1]), "status": "no-normal-but-best"}
            return None
        low, high, examples, best_prob = best_details
        return {"min": float(low), "max": float(high), "examples": [float(v) for v in examples], "best_prob": float(best_prob), "status": "normal-found"}
    
    @staticmethod
    def find_normal_range_multi(base_row: Dict, features: List[str]) -> Tuple[Optional[Dict], Dict, Optional[float], str]:
        usable = {feat: [float(bounds[0]), float(bounds[1])] for feat in features if (bounds := model_manager.numeric_feature_ranges.get(feat)) and np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and bounds[0] < bounds[1]}
        if not usable:
            return None, {}, None, "no-features"
        best_solution = None
        best_prob = None
        best_any_solution = None
        best_any_prob = None
        for _ in range(config.MAX_ITER_MULTI):
            trial = base_row.copy()
            mids = {}
            for feat, (low, high) in usable.items():
                mid = (low + high) / 2.0
                mids[feat] = mid
                trial[feat] = mid
            pred, prob = predictor.predict_single(trial, ignore_tryshot=True)
            if best_any_prob is None or prob < best_any_prob:
                best_any_prob = float(prob)
                best_any_solution = {feat: float(val) for feat, val in mids.items()}
            if pred == 0 and (best_prob is None or prob < best_prob):
                best_prob = float(prob)
                best_solution = {feat: float(val) for feat, val in mids.items()}
            updated = False
            for feat, (low, high) in list(usable.items()):
                mid = mids[feat]
                new_range = [low, mid] if (pred == 0 and (mid - low) >= (high - mid)) or (pred != 0 and (mid - low) < (high - mid)) else [mid, high]
                if new_range != usable[feat]:
                    usable[feat] = new_range
                    updated = True
            if not updated:
                break
        final_ranges = {feat: tuple(bounds) for feat, bounds in usable.items()}
        if best_solution:
            return best_solution, final_ranges, best_prob, "normal-found"
        elif best_any_solution:
            return best_any_solution, final_ranges, best_any_prob, "no-normal-but-best"
        return None, final_ranges, None, "no-solution"
    
    @staticmethod
    def evaluate_categorical(base_row: Dict, feature: str, choices: List[str]) -> Dict:
        candidates = []
        best_any = None
        for value in choices:
            trial = base_row.copy()
            trial[feature] = value
            pred, prob = predictor.predict_single(trial, ignore_tryshot=True)
            if best_any is None or prob < best_any[1]:
                best_any = (value, float(prob))
            if pred == 0:
                candidates.append((value, float(prob)))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return {"values": [val for val, _ in candidates[:config.TOP_K_CATEGORICAL]], "best_prob": float(candidates[0][1]), "status": "normal-found"}
        return {"values": [best_any[0]] if best_any else [], "best_prob": float(best_any[1]) if best_any else None, "status": "no-normal-but-best"}
    
    @staticmethod
    def recommend_ranges(base_row: Dict, focus_features: List[str]) -> Dict:
        if not focus_features:
            return {}
        recommendations = {}
        best_prob = None
        numeric_targets = [f for f in focus_features if f in model_manager.ui_numeric_cols]
        categorical_targets = [f for f in focus_features if f in model_manager.ui_categorical_cols]
        numeric_ranges = {f: model_manager.numeric_feature_ranges[f] for f in numeric_targets if f in model_manager.numeric_feature_ranges}
        if len(numeric_ranges) >= 2:
            solution, final_ranges, prob_multi, status = RecommendationEngine.find_normal_range_multi(base_row, list(numeric_ranges.keys()))
            if solution:
                for feat, mid in solution.items():
                    if bounds := final_ranges.get(feat, numeric_ranges.get(feat)):
                        recommendations[feat] = {"type": "numeric", "min": float(bounds[0]), "max": float(bounds[1]), "examples": [float(mid)], "method": "binary_multi", "status": status}
                if prob_multi is not None:
                    best_prob = prob_multi
        for feat, bounds in numeric_ranges.items():
            if details := RecommendationEngine.find_normal_range_single(base_row, feat, bounds):
                record = recommendations.get(feat, {"type": "numeric"})
                record.update({"min": details["min"], "max": details["max"], "examples": details.get("examples", [])[:3], "status": details.get("status", "normal-found")})
                if record.get("method") != "binary_multi":
                    record["method"] = "binary_search"
                recommendations[feat] = record
                if details.get("best_prob") is not None:
                    best_prob = details["best_prob"] if best_prob is None else min(best_prob, details["best_prob"])
        for feat in categorical_targets:
            if (meta := model_manager.input_metadata.get(feat)) and meta.get("choices"):
                if result := RecommendationEngine.evaluate_categorical(base_row, feat, meta["choices"]):
                    recommendations[feat] = {"type": "categorical", "values": result["values"], "status": result.get("status", "normal-found")}
                    if result.get("best_prob") is not None:
                        best_prob = result["best_prob"] if best_prob is None else min(best_prob, result["best_prob"])
        if best_prob is not None:
            recommendations["best_probability"] = float(best_prob)
        return recommendations

recommendation_engine = RecommendationEngine()

# ========== 통합 예측 함수 ==========
def predict_with_shap(row_dict: Dict) -> Optional[Dict]:
    if not model_manager.loaded:
        return None
    try:
        input_df = preprocessor.build_input_dataframe(row_dict)
        feature_matrix = preprocessor.prepare_feature_matrix(input_df)
        if feature_matrix is None:
            return None
        iteration = model_manager.get_booster_iteration()
        probs = model_manager.model.predict(feature_matrix, num_iteration=iteration) if iteration else model_manager.model.predict(feature_matrix)
        probability = float(probs[0])
        prediction = 1 if probability >= model_manager.threshold else 0
        forced_fail = False
        if (tryshot := str(row_dict.get("tryshot_signal", "")).upper()) == "D":
            if probability < model_manager.threshold:
                forced_fail = True
            prediction = 1
        contributions, shap_vector = shap_analyzer.compute_contributions(feature_matrix)
        explanation = shap_analyzer.build_explanation(contributions, shap_vector, row_dict)
        top_features = []
        if contributions:
            positive_items = [(f, c) for f, c in contributions.items() if c > 0]
            items = sorted(positive_items, key=lambda x: abs(x[1]), reverse=True)[:config.TOP_FEATURES_COUNT] if positive_items else sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:config.TOP_FEATURES_COUNT]
            for feat, contrib in items:
                top_features.append({"name": feat, "label": COLUMN_NAMES_KR.get(feat, feat), "value": row_dict.get(feat, np.nan), "contribution": contrib})
        recommendations = recommendation_engine.recommend_ranges(row_dict, [item["name"] for item in top_features])
        return {"probability": probability, "prediction": prediction, "forced_fail": forced_fail, "contributions": contributions, "shap_vector": shap_vector, "explanation": explanation, "top_features": top_features, "recommendations": recommendations, "input_row": row_dict}
    except Exception as e:
        logger.error(f"SHAP 예측 실패: {e}")
        return None

# ========== Shiny UI ==========
tab_ui = ui.page_fluid(
    ui.div(
        ui.div(ui.output_ui("tab_log_stats"), style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
        ui.div(ui.accordion(ui.accordion_panel(ui.HTML('<i class="fa-solid fa-exclamation-circle"></i> 누적 데이터 (불량)'), ui.output_ui("tab_log_table_defect_wrapper"), value="defect_panel"), id="data_accordion_1", open=True, multiple=True), style="background:#fff;border-radius:16px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
        ui.div(ui.div(ui.HTML('<i class="fa-solid fa-chart-bar"></i> SHAP 변수 영향도 측정'), style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #2A2D30"), ui.output_ui("shap_info_message"), ui.div(ui.div(ui.output_plot("shap_waterfall_plot", height="550px"), style="flex:7;min-width:0"), ui.div(ui.output_ui("shap_analysis_details"), style="flex:3;min-width:0;padding-left:20px;max-height:550px;overflow-y:auto"), style="display:flex;gap:20px;align-items:flex-start"), style="background:#fff;border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
        ui.div(ui.div(ui.HTML('<i class="fa-solid fa-industry"></i> 공정 프로세스 모니터링'), style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #2A2D30"), ui.output_ui("process_diagram"), style="background:#fff;border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);min-height:650px"),
        ui.div(ui.accordion(ui.accordion_panel(ui.HTML('<i class="fa-solid fa-table-list"></i> 누적 데이터 (전체)'), ui.output_ui("tab_log_table_all_wrapper"), value="all_panel"), id="data_accordion_2", open=False, multiple=True), style="background:#fff;border-radius:16px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
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
            predictions = predictor.predict_batch(df)
            temp_df = df.copy()
            temp_df['passorfail'] = predictions
            range_updater.update_ranges(temp_df)
            metadata_creator.create_metadata(df)
            return ui.div(ui.div(ui.div(ui.HTML(f'<i class="fa-solid fa-list-ol"></i> 총 데이터 행: <strong>{len(df):,}</strong>'), style="font-weight:600;font-size:16px;color:#2c3e50"), ui.div(ui.HTML(f'<i class="fa-solid fa-memory"></i> 메모리 사용량: <strong>{df.memory_usage(deep=True).sum() / 1024:.2f} KB</strong>'), style="font-weight:600;font-size:16px;color:#2c3e50"), ui.div(ui.HTML(f'<i class="fa-solid fa-exclamation-triangle"></i> 불량 건수: <strong>{int(predictions.sum()):,}</strong>'), style="font-weight:600;font-size:16px;color:#e74c3c"), style="display:flex;gap:360px;align-items:center;justify-content:flex-start"))
        except Exception as e:
            logger.error(f"통계 계산 오류: {e}")
            return ui.div("통계 계산 오류", style="color:#dc3545")
    
    @output
    @render.ui
    def tab_log_table_all_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...", style="text-align:center;padding:40px;color:#666"), style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")
        return ui.div(ui.output_data_frame("tab_log_table_all"), style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:300px")
    
    @output
    @render.data_frame
    def tab_log_table_all():
        try:
            df = shared_df.get()
            if df.empty:
                return render.DataGrid(pd.DataFrame(), height="300px", width="100%")
            result = df.copy()
            result['passorfail'] = predictor.predict_batch(result)
            result = result.drop(columns=config.DROP_COLUMNS, errors='ignore').tail(5).copy()
            if len(result) < 5:
                result = pd.concat([result, pd.DataFrame([[None] * len(result.columns)] * (5 - len(result)), columns=result.columns)], ignore_index=True)
            return render.DataGrid(result, height="300px", width="100%", filters=False, row_selection_mode="none")
        except Exception as e:
            logger.error(f"테이블 렌더링 오류: {e}")
            return render.DataGrid(pd.DataFrame({"오류": [str(e)]}), height="300px", width="100%")
    
    @output
    @render.ui
    def tab_log_table_defect_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...", style="text-align:center;padding:40px;color:#666"), style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")
        return ui.div(ui.output_data_frame("tab_log_table_defect"), style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:300px")
    
    @output
    @render.data_frame
    def tab_log_table_defect():
        try:
            df = shared_df.get()
            if df.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["데이터 없음"]}), height="300px", width="100%")
            result = df.copy()
            result['passorfail'] = predictor.predict_batch(result)
            defect_only = result[result['passorfail'] == 1].drop(columns=config.DROP_COLUMNS, errors='ignore').tail(5).copy()
            if defect_only.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["✅ 불량 없음"]}), height="300px", width="100%")
            if len(defect_only) < 5:
                defect_only = pd.concat([defect_only, pd.DataFrame([[None] * len(defect_only.columns)] * (5 - len(defect_only)), columns=defect_only.columns)], ignore_index=True)
            return render.DataGrid(defect_only, height="300px", width="100%", filters=False, row_selection_mode="single")
        except Exception as e:
            logger.error(f"불량 테이블 오류: {e}")
            return render.DataGrid(pd.DataFrame({"오류": [str(e)]}), height="300px", width="100%")
    
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
            result['passorfail'] = predictor.predict_batch(result)
            range_updater.update_ranges(result)
            defect_only = result[result['passorfail'] == 1].copy()
            if defect_only.empty or (idx := list(selected)[0]) >= len(defect_only):
                return
            row = defect_only.iloc[idx]
            selected_row_data.set(row)
            analysis_result.set(predict_with_shap(row.to_dict()))
        except Exception as e:
            logger.error(f"행 선택 오류: {e}")
    
    @output
    @render.ui
    def shap_info_message():
        if (result := analysis_result.get()) is None:
            return ui.div()
        prob = result['probability']
        pred_label = "불량" if result['prediction'] == 1 else "정상"
        pred_color = "#dc3545" if pred_label == "불량" else "#28a745"
        forced_msg = '<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:12px;margin-top:10px;border-radius:4px;">⚠️ <strong>tryshot_signal 규칙 적용</strong><br><span style="font-size:12px;color:#6c757d;">※ 추천 탐색 중에는 강제불량을 일시 무시합니다.</span></div>' if result['forced_fail'] else ''
        return ui.HTML(f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-bottom:20px;"><div style="font-size:24px;font-weight:700;color:{pred_color};margin-bottom:15px;">{pred_label}</div><div style="font-size:15px;margin-bottom:8px;">불량 확률: <strong style="font-size:18px;color:#dc3545;">{prob:.4f}</strong> (임계값: {model_manager.threshold:.4f})</div>{forced_msg}</div>')
    
    @output
    @render.plot
    def shap_waterfall_plot():
        result = analysis_result.get()
        fig, ax = plt.subplots(figsize=(8, 6))
        if result is None or (explanation := result.get('explanation')) is None:
            ax.axis("off")
            if result is not None:
                ax.text(0.5, 0.5, "SHAP 생성 실패", ha="center", va="center", fontsize=14, color="#dc3545")
            plt.tight_layout()
            return fig
        try:
            plt.close('all')
            setup_korean_font()
            shap.plots.waterfall(explanation, max_display=20, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Plot 생성 오류: {e}")
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")
            ax.text(0.5, 0.5, "Plot 생성 실패", ha="center", va="center", fontsize=12, color="#dc3545")
            plt.tight_layout()
            return fig
    
    @output
    @render.ui
    def shap_analysis_details():
        if (result := analysis_result.get()) is None or not (top_features := result.get('top_features', [])):
            return ui.div()
        recommendations = result.get('recommendations', {})
        items = []
        for rank, item in enumerate(top_features, 1):
            feat_name = item["name"]
            val = format_value(item.get("value"))
            contrib = item.get("contribution", 0.0)
            direction = "위험 증가" if contrib > 0 else "위험 감소"
            color = "#dc3545" if contrib > 0 else "#28a745"
            item_html = f'<li style="margin-bottom:14px;"><div style="display:flex;align-items:center;gap:6px;"><span style="background:#2A2D30;color:white;border-radius:50%;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">{rank}</span><strong style="font-size:14px;">{item["label"]}</strong></div><div style="margin-left:28px;margin-top:3px;"><span style="color:#6c757d;font-size:12px;">현재 값: {val}</span><br>'
            if rec := recommendations.get(feat_name, {}):
                if rec.get("type") == "numeric":
                    min_v, max_v = format_value(rec.get("min")), format_value(rec.get("max"))
                    status = rec.get("status")
                    item_html += f'<span style="color:{"#856404" if status == "no-normal-but-best" else "#28a745"};font-weight:600;font-size:12px;">{"• 정상 전환 불가, 확률 최소화 후보" if status == "no-normal-but-best" else "✓ 정상 전환 추천"}: {min_v} ~ {max_v}</span><br>'
                elif rec.get("type") == "categorical":
                    values = ", ".join(rec.get("values", []))
                    status = rec.get("status")
                    item_html += f'<span style="color:{"#856404" if status == "no-normal-but-best" else "#28a745"};font-weight:600;font-size:12px;">{"• 정상 전환 불가, 확률 최소화 후보" if status == "no-normal-but-best" else "✓ 정상 전환 추천 값"}: {values}</span><br>'
            item_html += f'<span style="font-size:12px;">SHAP 기여도: </span><span style="color:{color};font-weight:600;font-size:13px;">{contrib:+.4f}</span> <span style="color:#6c757d;font-size:11px;">({direction})</span></div></li>'
            items.append(item_html)
        prob_html = f'<div style="background:#d4edda;border-left:4px solid #28a745;padding:10px;margin-top:14px;border-radius:4px;"><strong style="font-size:13px;">📈 추천 적용 시 예상 불량 확률(최소): {best_prob:.4f}</strong></div>' if (best_prob := recommendations.get("best_probability")) is not None else ""
        return ui.HTML(f'<div style="background:#f8f9fa;padding:18px;border-radius:8px;height:100%;"><div style="font-weight:600;font-size:15px;margin-bottom:14px;border-bottom:2px solid #dee2e6;padding-bottom:7px;">📊 불량 영향 상위 변수 및 정상/최소확률 전환 권고</div><ul style="padding-left:18px;margin:0;">{"".join(items)}</ul>{prob_html}</div>')
    
    @output
    @render.ui
    def process_diagram():
        if (result := analysis_result.get()) is None:
            return ui.div(ui.div("⏳ 불량 행을 선택하면 공정 프로세스 상태가 표시됩니다.", style="text-align:center;padding:60px;color:#999;font-size:14px"), style="position:relative;width:100%;height:600px;background:#f8f9fa;border-radius:8px")
        contributions = result.get('contributions', {})
        legend_html = '<div style="background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:16px;border:1px solid #dee2e6;display:flex;gap:20px;align-items:center;justify-content:center;"><div style="font-weight:600;color:#2c3e50;margin-right:10px;">신호등 색상 기준:</div><div style="display:flex;align-items:center;gap:6px;"><div style="width:14px;height:14px;border-radius:50%;background:#dc3545;border:2px solid #333;"></div><span style="font-size:13px;color:#2c3e50;">빨간불 (SHAP > 0.15)</span></div><div style="display:flex;align-items:center;gap:6px;"><div style="width:14px;height:14px;border-radius:50%;background:#ffc107;border:2px solid #333;"></div><span style="font-size:13px;color:#2c3e50;">노란불 (SHAP > 0.05)</span></div><div style="display:flex;align-items:center;gap:6px;"><div style="width:14px;height:14px;border-radius:50%;background:#28a745;border:2px solid #333;"></div><span style="font-size:13px;color:#2c3e50;">초록불 (정상)</span></div></div>'
        image_base64 = get_process_image_base64()
        background_style = f"background:url('{image_base64}') center/contain no-repeat #f8f9fa" if image_base64 else "background:#f8f9fa"
        accordion_style = '<style>.process-accordion{position:absolute;background:rgba(255,255,255,0.95);border:2px solid #2c3e50;border-radius:8px;min-width:200px;box-shadow:0 4px 12px rgba(0,0,0,0.15);overflow:hidden}.process-accordion summary{font-weight:700;font-size:13px;color:#2c3e50;padding:12px;text-align:center;cursor:pointer;list-style:none;user-select:none;border-bottom:2px solid #2c3e50;background:rgba(255,255,255,0.98)}.process-accordion summary::-webkit-details-marker{display:none}.process-accordion summary::after{content:"▼";float:right;font-size:10px;transition:transform 0.2s}.process-accordion[open] summary::after{transform:rotate(-180deg)}.process-accordion summary:hover{background:#f0f0f0}.process-accordion-content{padding:12px}</style>'
        process_boxes = []
        for process_name, process_info in PROCESS_MAPPING.items():
            variable_rows = []
            for var in process_info["variables"]:
                var_label = COLUMN_NAMES_KR.get(var, var)
                shap_value = contributions.get(var, 0.0)
                light_color = "#dc3545" if shap_value > SHAP_THRESHOLD_CRITICAL else ("#ffc107" if shap_value > SHAP_THRESHOLD_WARNING else "#28a745")
                status_text = "위험" if light_color == "#dc3545" else ("경고" if light_color == "#ffc107" else "정상")
                variable_rows.append(f'<div style="display:flex;align-items:center;justify-content:space-between;padding:4px 8px;margin:3px 0;background:#fff;border-radius:4px;"><span style="font-size:11px;color:#2c3e50;flex:1">{var_label}</span><div style="width:16px;height:16px;border-radius:50%;background:{light_color};border:2px solid #333;box-shadow:0 0 8px {light_color}80;margin-left:8px" title="{var_label}: {status_text} (SHAP: {shap_value:.4f})"></div></div>')
            position = process_info["position"]
            process_boxes.append(f'<details class="process-accordion" open style="top:{position["top"]};left:{position["left"]};transform:translate(-50%,-50%);"><summary>{process_name}</summary><div class="process-accordion-content">{"".join(variable_rows)}</div></details>')
        return ui.HTML(f'{accordion_style}{legend_html}<div style="position:relative;width:100%;height:600px;{background_style};border:1px solid #e0e0e0;border-radius:8px">{"".join(process_boxes)}</div>')