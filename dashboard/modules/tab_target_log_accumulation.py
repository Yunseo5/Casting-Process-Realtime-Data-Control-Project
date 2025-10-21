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

# ========== 로깅 설정 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 상수 정의 ==========
@dataclass
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    MODEL_PATH: Path = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"
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

# ========== 한글 폰트 설정 ==========
def setup_korean_font() -> None:
    """유니코드 마이너스를 포함한 한글 폰트 설정"""
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    korean_fonts = ['Noto Sans KR', 'Noto Sans CJK KR', 'NanumGothic', 'AppleGothic', 'Malgun Gothic']
    
    chosen = next((f for f in korean_fonts if f in available_fonts), None)
    plt.rcParams['font.family'] = [chosen, 'DejaVu Sans'] if chosen else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = True
    logger.info(f"폰트 설정: {chosen or 'DejaVu Sans'}")

setup_korean_font()

# ========== 모델 클래스 ==========
class DefectPredictionModel:
    """불량 예측 모델 관리 클래스"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.loaded = False
        self.model = None
        self.scaler = None
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.threshold = config.DEFAULT_THRESHOLD
        self.explainer = None
        
        # 모델 스키마
        self.model_numeric_cols: List[str] = []
        self.model_categorical_cols: List[str] = []
        self.model_required_cols: List[str] = []
        
        # UI 스키마
        self.ui_numeric_cols: List[str] = []
        self.ui_categorical_cols: List[str] = []
        
        # 메타데이터
        self.numeric_feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.input_metadata: Dict[str, Any] = {}
        
        # SHAP 매핑
        self.numeric_index_map: Dict[str, int] = {}
        self.ohe_feature_slices: Dict[str, Tuple[int, int]] = {}
        
        self._load_model()
    
    def _load_model(self) -> None:
        """모델 로드 및 초기화"""
        try:
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.scaler = artifact.get("scaler")
            self.ordinal_encoder = artifact.get("ordinal_encoder")
            self.onehot_encoder = artifact.get("onehot_encoder")
            
            # 임계값 설정
            threshold = artifact.get("operating_threshold")
            if threshold is None:
                self.threshold = config.DEFAULT_THRESHOLD
                logger.warning(f"모델 파일에 임계값 없음 → {config.DEFAULT_THRESHOLD}로 설정")
            else:
                self.threshold = float(threshold)
                if abs(self.threshold - 0.5) < 0.001:
                    logger.warning(f"임계값이 0.5로 설정되어 있음 → {config.DEFAULT_THRESHOLD}로 변경")
                    self.threshold = config.DEFAULT_THRESHOLD
            
            # 스키마 초기화
            self._initialize_schema()
            self._initialize_shap()
            
            self.loaded = True
            logger.info(f"모델 로드 성공 (임계값: {self.threshold})")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.loaded = False
    
    def _initialize_schema(self) -> None:
        """스키마 초기화"""
        if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
            self.model_numeric_cols = list(self.scaler.feature_names_in_)
        
        if self.ordinal_encoder and hasattr(self.ordinal_encoder, 'feature_names_in_'):
            self.model_categorical_cols = list(self.ordinal_encoder.feature_names_in_)
        
        self.model_required_cols = self.model_numeric_cols + self.model_categorical_cols
        self.ui_numeric_cols = self.model_numeric_cols.copy()
        self.ui_categorical_cols = self.model_categorical_cols.copy()
    
    def _initialize_shap(self) -> None:
        """SHAP explainer 초기화"""
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
        """모델 iteration 반환"""
        iteration = getattr(self.model, "best_iteration", None)
        if not iteration:
            current_iteration = getattr(self.model, "current_iteration", None)
            if callable(current_iteration):
                iteration = current_iteration()
        return int(iteration) if iteration else None

# 전역 모델 인스턴스
model_manager = DefectPredictionModel(config.MODEL_PATH)

# ========== 유틸리티 함수 ==========
def format_value(value: Any) -> str:
    """값 포맷팅"""
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)

def safe_numeric_conversion(series: pd.Series) -> pd.Series:
    """안전한 수치형 변환"""
    return pd.to_numeric(series, errors='coerce')

# ========== 데이터 전처리 ==========
class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    @staticmethod
    def build_input_dataframe(row_dict: Dict[str, Any]) -> pd.DataFrame:
        """입력 데이터프레임 생성"""
        data = {col: row_dict.get(col) for col in model_manager.model_required_cols}
        input_df = pd.DataFrame([data], columns=model_manager.model_required_cols)
        
        if model_manager.model_numeric_cols:
            input_df[model_manager.model_numeric_cols] = input_df[model_manager.model_numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )
        
        if model_manager.model_categorical_cols:
            input_df[model_manager.model_categorical_cols] = (
                input_df[model_manager.model_categorical_cols].fillna("UNKNOWN").astype(str)
            )
        
        return input_df
    
    @staticmethod
    def prepare_feature_matrix(input_df: pd.DataFrame) -> Optional[np.ndarray]:
        """특성 행렬 준비"""
        if not model_manager.loaded:
            return None
        
        try:
            arrays = []
            
            if model_manager.model_numeric_cols and model_manager.scaler:
                arrays.append(model_manager.scaler.transform(
                    input_df[model_manager.model_numeric_cols].astype(float)
                ))
            
            if (model_manager.model_categorical_cols and 
                model_manager.ordinal_encoder and 
                model_manager.onehot_encoder):
                cat_ord = model_manager.ordinal_encoder.transform(
                    input_df[model_manager.model_categorical_cols]
                ).astype(int)
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
    """불량 예측 클래스"""
    
    @staticmethod
    def predict_batch(df: pd.DataFrame) -> np.ndarray:
        """배치 예측"""
        if not model_manager.loaded or df.empty:
            return np.zeros(len(df), dtype=int)
        
        try:
            X = df.drop(columns=config.DROP_COLUMNS + ['passorfail'], errors='ignore').copy()
            
            # 결측 처리
            for col in model_manager.model_required_cols:
                if col not in X.columns:
                    X[col] = 0.0 if col in model_manager.model_numeric_cols else 'UNKNOWN'
            
            X = X[model_manager.model_required_cols].copy()
            
            # 변환
            arrays = []
            if model_manager.model_numeric_cols and model_manager.scaler:
                X_num = X[model_manager.model_numeric_cols].fillna(0.0)
                arrays.append(model_manager.scaler.transform(X_num))
            
            if (model_manager.model_categorical_cols and 
                model_manager.ordinal_encoder and 
                model_manager.onehot_encoder):
                X_cat = X[model_manager.model_categorical_cols].fillna('UNKNOWN')
                X_cat_ord = model_manager.ordinal_encoder.transform(X_cat).astype(int)
                arrays.append(model_manager.onehot_encoder.transform(X_cat_ord))
            
            if not arrays:
                return np.zeros(len(df), dtype=int)
            
            X_final = np.hstack(arrays)
            
            # 예측
            iteration = model_manager.get_booster_iteration()
            probs = (model_manager.model.predict(X_final, num_iteration=iteration) 
                    if iteration else model_manager.model.predict(X_final))
            predictions = (probs >= model_manager.threshold).astype(int)
            
            # tryshot 규칙
            if 'tryshot_signal' in df.columns:
                predictions[df['tryshot_signal'].to_numpy() == 'D'] = 1
            
            return predictions
        
        except Exception as e:
            logger.error(f"배치 예측 실패: {e}")
            return np.zeros(len(df), dtype=int)
    
    @staticmethod
    def predict_single(row_dict: Dict[str, Any], ignore_tryshot: bool = False) -> Tuple[int, float]:
        """단일 예측"""
        if not model_manager.loaded:
            return 0, 0.0
        
        try:
            input_df = preprocessor.build_input_dataframe(row_dict)
            feature_matrix = preprocessor.prepare_feature_matrix(input_df)
            
            if feature_matrix is None:
                return 0, 0.0
            
            iteration = model_manager.get_booster_iteration()
            probs = (model_manager.model.predict(feature_matrix, num_iteration=iteration) 
                    if iteration else model_manager.model.predict(feature_matrix))
            probability = float(probs[0])
            prediction = 1 if probability >= model_manager.threshold else 0
            
            if not ignore_tryshot:
                tryshot = row_dict.get("tryshot_signal")
                if tryshot and str(tryshot).upper() == "D":
                    prediction = 1
            
            return prediction, probability
        
        except Exception as e:
            logger.error(f"단일 예측 실패: {e}")
            return 0, 0.0

predictor = Predictor()

# ========== 범위 업데이트 ==========
class RangeUpdater:
    """수치형 특성 범위 업데이트 클래스"""
    
    @staticmethod
    def update_ranges(df: pd.DataFrame) -> None:
        """범위 업데이트"""
        if df.empty:
            logger.warning("빈 데이터프레임 - 범위 업데이트 건너뜀")
            return
        
        try:
            # 정상 데이터 추출
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
                
                q_low = series.quantile(config.PERCENTILE_LOW)
                q_high = series.quantile(config.PERCENTILE_HIGH)
                span = q_high - q_low
                
                if not np.isfinite(span) or span <= 0:
                    min_val = q_low * 0.9 if q_low != 0 else -1.0
                    max_val = q_low * 1.1 if q_low != 0 else 1.0
                else:
                    min_val = float(q_low - config.RANGE_BUFFER * span)
                    max_val = float(q_high + config.RANGE_BUFFER * span)
                
                model_manager.numeric_feature_ranges[col] = (min_val, max_val)
            
            # 기본 범위 설정
            if not model_manager.numeric_feature_ranges:
                for col in model_manager.ui_numeric_cols:
                    model_manager.numeric_feature_ranges[col] = (0.0, 100.0)
            
            logger.info(f"범위 업데이트 완료: {len(model_manager.numeric_feature_ranges)}개 변수")
        
        except Exception as e:
            logger.error(f"범위 업데이트 실패: {e}")
            for col in model_manager.ui_numeric_cols:
                model_manager.numeric_feature_ranges[col] = (0.0, 100.0)

range_updater = RangeUpdater()

# ========== 메타데이터 생성 ==========
class MetadataCreator:
    """입력 메타데이터 생성 클래스"""
    
    @staticmethod
    def create_metadata(df: pd.DataFrame) -> None:
        """메타데이터 생성"""
        metadata = {}
        
        # 휴리스틱 범주형 추출
        heuristic_categorical = MetadataCreator._extract_categorical_features(df)
        
        # 스키마 보정
        cat_set = set(model_manager.ui_categorical_cols) | set(heuristic_categorical)
        num_set = set(model_manager.ui_numeric_cols) - set(heuristic_categorical)
        
        # 범주형 메타데이터
        for col in cat_set:
            if col not in df.columns:
                continue
            values = sorted([str(v) for v in df[col].astype(str).dropna().unique()])
            if values:
                metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}
        
        # 수치형 메타데이터
        for col in num_set:
            if col not in df.columns:
                continue
            s = safe_numeric_conversion(df[col]).dropna()
            if len(s) == 0:
                continue
            vmin, vmax = float(s.quantile(config.PERCENTILE_LOW)), float(s.quantile(config.PERCENTILE_HIGH))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
            metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": float(s.median())}
        
        # 스키마 갱신
        model_manager.ui_categorical_cols = list(cat_set)
        model_manager.ui_numeric_cols = list(num_set)
        model_manager.input_metadata = metadata
        
        logger.info(f"메타데이터 생성 완료: 범주형 {len([m for m in metadata.values() if m['type'] == 'categorical'])}개, "
                   f"수치형 {len([m for m in metadata.values() if m['type'] == 'numeric'])}개")
    
    @staticmethod
    def _extract_categorical_features(df: pd.DataFrame) -> List[str]:
        """휴리스틱 범주형 특성 추출"""
        heuristic = []
        for col in df.columns:
            col_l = str(col).lower()
            if col_l in ["mold_code", "mold", "code", "model_code"]:
                heuristic.append(col)
            elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique(dropna=True) <= 20:
                heuristic.append(col)
        return list(dict.fromkeys(heuristic))

metadata_creator = MetadataCreator()

# ========== SHAP 분석 ==========
class SHAPAnalyzer:
    """SHAP 분석 클래스"""
    
    @staticmethod
    def compute_contributions(feature_matrix: np.ndarray) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """SHAP 기여도 계산"""
        if not model_manager.loaded or model_manager.explainer is None or feature_matrix is None:
            return None, None
        
        try:
            shap_values = model_manager.explainer.shap_values(feature_matrix)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            shap_vector = shap_values[0]
            contributions = {}
            
            # 수치형 기여도
            for feat, idx in model_manager.numeric_index_map.items():
                contributions[feat] = float(shap_vector[idx])
            
            # 범주형 기여도
            for feat, (start, end) in model_manager.ohe_feature_slices.items():
                contributions[feat] = float(np.sum(shap_vector[start:end])) if end > start else 0.0
            
            return contributions, shap_vector
        
        except Exception as e:
            logger.error(f"SHAP 기여도 계산 실패: {e}")
            return None, None
    
    @staticmethod
    def build_explanation(contributions: Dict, shap_vector: np.ndarray, input_row: Dict) -> Optional[Any]:
        """SHAP explanation 객체 생성"""
        if contributions is None or shap_vector is None:
            return None
        
        try:
            from shap import Explanation
            
            expected_value = (float(model_manager.explainer.expected_value[1]) 
                            if isinstance(model_manager.explainer.expected_value, (list, np.ndarray)) 
                            else float(model_manager.explainer.expected_value))
            
            shap_values = np.array([float(contributions.get(col, 0.0)) 
                                   for col in model_manager.model_required_cols], dtype=float)
            feature_values = np.array(SHAPAnalyzer._extract_feature_values(input_row), dtype=object)
            feature_names = [COLUMN_NAMES_KR.get(col, col) for col in model_manager.model_required_cols]
            
            return Explanation(values=shap_values, base_values=expected_value, 
                             data=feature_values, feature_names=feature_names)
        
        except Exception as e:
            logger.error(f"SHAP explanation 생성 실패: {e}")
            return None
    
    @staticmethod
    def _extract_feature_values(row_dict: Dict) -> List:
        """특성 값 추출"""
        values = []
        for col in model_manager.model_required_cols:
            val = row_dict.get(col, np.nan)
            if isinstance(val, (list, tuple)):
                val = val[0] if len(val) > 0 else np.nan
            if isinstance(val, (pd.Timestamp, pd.Timedelta)):
                val = str(val)
            values.append(np.nan if pd.isna(val) else val)
        return values

shap_analyzer = SHAPAnalyzer()

# ========== 추천 시스템 ==========
class RecommendationEngine:
    """정상 범위 추천 엔진"""
    
    @staticmethod
    def find_normal_range_single(base_row: Dict, feature: str, bounds: Tuple[float, float]) -> Optional[Dict]:
        """단일 변수 이진 탐색"""
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
                return {
                    "min": float(bounds[0]), "max": float(bounds[1]),
                    "examples": [float(best_overall[0])],
                    "best_prob": float(best_overall[1]),
                    "status": "no-normal-but-best"
                }
            return None
        
        low, high, examples, best_prob = best_details
        return {
            "min": float(low), "max": float(high),
            "examples": [float(v) for v in examples],
            "best_prob": float(best_prob),
            "status": "normal-found"
        }
    
    @staticmethod
    def find_normal_range_multi(base_row: Dict, features: List[str]) -> Tuple[Optional[Dict], Dict, Optional[float], str]:
        """다변수 동시 이진 탐색"""
        usable = {}
        for feat in features:
            bounds = model_manager.numeric_feature_ranges.get(feat)
            if bounds and np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and bounds[0] < bounds[1]:
                usable[feat] = [float(bounds[0]), float(bounds[1])]
        
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
            
            # 범위 축소
            updated = False
            for feat, (low, high) in list(usable.items()):
                mid = mids[feat]
                new_range = ([low, mid] if (mid - low) >= (high - mid) else [mid, high]) if pred == 0 else ([mid, high] if (mid - low) >= (high - mid) else [low, mid])
                
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
        """범주형 변수 평가"""
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
            return {
                "values": [val for val, _ in candidates[:config.TOP_K_CATEGORICAL]],
                "best_prob": float(candidates[0][1]),
                "status": "normal-found"
            }
        
        return {
            "values": [best_any[0]] if best_any else [],
            "best_prob": float(best_any[1]) if best_any else None,
            "status": "no-normal-but-best"
        }
    
    @staticmethod
    def recommend_ranges(base_row: Dict, focus_features: List[str]) -> Dict:
        """추천 범위 생성"""
        if not focus_features:
            return {}
        
        recommendations = {}
        best_prob = None
        
        numeric_targets = [f for f in focus_features if f in model_manager.ui_numeric_cols]
        categorical_targets = [f for f in focus_features if f in model_manager.ui_categorical_cols]
        
        numeric_ranges = {f: model_manager.numeric_feature_ranges[f] 
                         for f in numeric_targets if f in model_manager.numeric_feature_ranges}
        
        # 다변수 최적화
        if len(numeric_ranges) >= 2:
            solution, final_ranges, prob_multi, status = RecommendationEngine.find_normal_range_multi(
                base_row, list(numeric_ranges.keys())
            )
            if solution:
                for feat, mid in solution.items():
                    bounds = final_ranges.get(feat, numeric_ranges.get(feat))
                    if bounds:
                        recommendations[feat] = {
                            "type": "numeric", "min": float(bounds[0]), "max": float(bounds[1]),
                            "examples": [float(mid)], "method": "binary_multi", "status": status
                        }
                if prob_multi is not None:
                    best_prob = prob_multi
        
        # 단일 변수 탐색
        for feat, bounds in numeric_ranges.items():
            details = RecommendationEngine.find_normal_range_single(base_row, feat, bounds)
            if details:
                record = recommendations.get(feat, {"type": "numeric"})
                record.update({
                    "min": details["min"], "max": details["max"],
                    "examples": details.get("examples", [])[:3],
                    "status": details.get("status", "normal-found")
                })
                if record.get("method") != "binary_multi":
                    record["method"] = "binary_search"
                recommendations[feat] = record
                
                if details.get("best_prob") is not None:
                    best_prob = details["best_prob"] if best_prob is None else min(best_prob, details["best_prob"])
        
        # 범주형 평가
        for feat in categorical_targets:
            meta = model_manager.input_metadata.get(feat)
            if meta and meta.get("choices"):
                result = RecommendationEngine.evaluate_categorical(base_row, feat, meta["choices"])
                if result:
                    recommendations[feat] = {
                        "type": "categorical",
                        "values": result["values"],
                        "status": result.get("status", "normal-found")
                    }
                    if result.get("best_prob") is not None:
                        best_prob = result["best_prob"] if best_prob is None else min(best_prob, result["best_prob"])
        
        if best_prob is not None:
            recommendations["best_probability"] = float(best_prob)
        
        return recommendations

recommendation_engine = RecommendationEngine()

# ========== 통합 예측 함수 ==========
def predict_with_shap(row_dict: Dict) -> Optional[Dict]:
    """SHAP 예측 및 추천"""
    if not model_manager.loaded:
        return None
    
    try:
        input_df = preprocessor.build_input_dataframe(row_dict)
        feature_matrix = preprocessor.prepare_feature_matrix(input_df)
        if feature_matrix is None:
            return None
        
        # 예측
        iteration = model_manager.get_booster_iteration()
        probs = (model_manager.model.predict(feature_matrix, num_iteration=iteration) 
                if iteration else model_manager.model.predict(feature_matrix))
        probability = float(probs[0])
        prediction = 1 if probability >= model_manager.threshold else 0
        
        # tryshot 규칙
        forced_fail = False
        tryshot = row_dict.get("tryshot_signal")
        if tryshot and str(tryshot).upper() == "D":
            if probability < model_manager.threshold:
                forced_fail = True
            prediction = 1
        
        # SHAP 분석
        contributions, shap_vector = shap_analyzer.compute_contributions(feature_matrix)
        explanation = shap_analyzer.build_explanation(contributions, shap_vector, row_dict)
        
        # 상위 특성 추출
        top_features = []
        if contributions:
            positive_items = [(f, c) for f, c in contributions.items() if c > 0]
            items = (sorted(positive_items, key=lambda x: abs(x[1]), reverse=True)[:config.TOP_FEATURES_COUNT] 
                    if positive_items else 
                    sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:config.TOP_FEATURES_COUNT])
            
            for feat, contrib in items:
                top_features.append({
                    "name": feat,
                    "label": COLUMN_NAMES_KR.get(feat, feat),
                    "value": row_dict.get(feat, np.nan),
                    "contribution": contrib
                })
        
        # 추천
        recommendations = recommendation_engine.recommend_ranges(
            row_dict, [item["name"] for item in top_features]
        )
        
        return {
            "probability": probability, "prediction": prediction, "forced_fail": forced_fail,
            "contributions": contributions, "shap_vector": shap_vector, "explanation": explanation,
            "top_features": top_features, "recommendations": recommendations, "input_row": row_dict
        }
    
    except Exception as e:
        logger.error(f"SHAP 예측 실패: {e}")
        return None

# ========== Shiny UI ==========
tab_ui = ui.page_fluid(
    ui.div(
        ui.div(ui.output_ui("tab_log_stats"),
               style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
        
        ui.div(
            ui.accordion(
                ui.accordion_panel(
                    ui.HTML('<i class="fa-solid fa-exclamation-circle"></i> 누적 데이터 (불량)'),
                    ui.output_ui("tab_log_table_defect_wrapper"),
                    value="defect_panel"
                ),
                id="data_accordion_1",
                open=True,
                multiple=True
            ),
            style="background:#fff;border-radius:16px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"
        ),
        
        ui.div(
            ui.div(ui.HTML('<i class="fa-solid fa-chart-bar"></i> SHAP 변수 영향도 측정'),
                   style="font-size:18px;font-weight:700;color:#2A2D30;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid #2A2D30"),
            ui.output_ui("shap_info_message"),
            ui.div(
                ui.div(ui.output_plot("shap_waterfall_plot", height="550px"),
                       style="flex:6;min-width:0"),
                ui.div(ui.output_ui("shap_analysis_details"),
                       style="flex:4;min-width:0;padding-left:20px;max-height:550px;overflow-y:auto"),
                style="display:flex;gap:20px;align-items:flex-start"),
            style="background:#fff;border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"),
        
        ui.div(
            ui.accordion(
                ui.accordion_panel(
                    ui.HTML('<i class="fa-solid fa-table-list"></i> 누적 데이터 (전체)'),
                    ui.output_ui("tab_log_table_all_wrapper"),
                    value="all_panel"
                ),
                id="data_accordion_2",
                open=False,
                multiple=True
            ),
            style="background:#fff;border-radius:16px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)"
        ),
        
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
            
            memory_usage = f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            defect_count = int(predictions.sum())
            
            return ui.div(
                ui.div(
                    ui.div(ui.HTML(f'<i class="fa-solid fa-list-ol"></i> 총 데이터 행: <strong>{len(df):,}</strong>'),
                           style="font-weight:600;font-size:16px;color:#2c3e50"),
                    ui.div(ui.HTML(f'<i class="fa-solid fa-memory"></i> 메모리 사용량: <strong>{memory_usage}</strong>'),
                           style="font-weight:600;font-size:16px;color:#2c3e50"),
                    ui.div(ui.HTML(f'<i class="fa-solid fa-exclamation-triangle"></i> 불량 건수: <strong>{defect_count:,}</strong>'),
                           style="font-weight:600;font-size:16px;color:#e74c3c"),
                    style="display:flex;gap:360px;align-items:center;justify-content:flex-start"))
        except Exception as e:
            logger.error(f"통계 계산 오류: {e}")
            return ui.div("통계 계산 오류", style="color:#dc3545")
    
    @output
    @render.ui
    def tab_log_table_all_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...", 
                                style="text-align:center;padding:40px;color:#666"),
                         style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")
        return ui.div(ui.output_data_frame("tab_log_table_all"),
                     style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:300px")
    
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
            
            # 5개 미만이면 빈 행으로 패딩
            if len(result) < 5:
                pad = pd.DataFrame([[None] * len(result.columns)] * (5 - len(result)), columns=result.columns)
                result = pd.concat([result, pad], ignore_index=True)
            
            return render.DataGrid(result, height="300px", width="100%", 
                                  filters=False, row_selection_mode="none")
        except Exception as e:
            logger.error(f"테이블 렌더링 오류: {e}")
            return render.DataGrid(pd.DataFrame({"오류": [str(e)]}), height="300px", width="100%")
    
    @output
    @render.ui
    def tab_log_table_defect_wrapper():
        df = shared_df.get()
        if df.empty:
            return ui.div(ui.div("⏳ 데이터를 불러오는 중...",
                                style="text-align:center;padding:40px;color:#666"),
                         style="width:100%;border:1px solid #e0e0e0;border-radius:8px;background:#f9f9f9")
        return ui.div(ui.output_data_frame("tab_log_table_defect"),
                     style="width:100%;overflow:auto;border:1px solid #e0e0e0;border-radius:8px;height:300px")
    
    @output
    @render.data_frame
    def tab_log_table_defect():
        try:
            df = shared_df.get()
            if df.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["데이터 없음"]}), 
                                      height="300px", width="100%")
            
            result = df.copy()
            result['passorfail'] = predictor.predict_batch(result)
            defect_only = result[result['passorfail'] == 1].copy()
            defect_only = defect_only.drop(columns=config.DROP_COLUMNS, errors='ignore').tail(5).copy()
            
            if defect_only.empty:
                return render.DataGrid(pd.DataFrame({"메시지": ["✅ 불량 없음"]}),
                                      height="300px", width="100%")
            
            # 5개 미만이면 빈 행으로 패딩
            if len(defect_only) < 5:
                pad = pd.DataFrame([[None] * len(defect_only.columns)] * (5 - len(defect_only)), 
                                  columns=defect_only.columns)
                defect_only = pd.concat([defect_only, pad], ignore_index=True)
            
            return render.DataGrid(defect_only, height="300px", width="100%",
                                  filters=False, row_selection_mode="single")
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
            if defect_only.empty:
                return
            
            idx = list(selected)[0]
            if idx >= len(defect_only):
                return
            
            row = defect_only.iloc[idx]
            selected_row_data.set(row)
            
            shap_result = predict_with_shap(row.to_dict())
            analysis_result.set(shap_result)
        
        except Exception as e:
            logger.error(f"행 선택 오류: {e}")
    
    @output
    @render.ui
    def shap_info_message():
        result = analysis_result.get()
        if result is None:
            return ui.div()
        
        prob = result['probability']
        pred_label = "불량" if result['prediction'] == 1 else "정상"
        pred_color = "#dc3545" if pred_label == "불량" else "#28a745"
        
        forced_msg = ''
        if result['forced_fail']:
            forced_msg = (
                '<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:12px;'
                'margin-top:10px;border-radius:4px;">⚠️ <strong>tryshot_signal 규칙 적용</strong>'
                '<br><span style="font-size:12px;color:#6c757d;">※ 추천 탐색 중에는 강제불량을 일시 무시합니다.</span></div>'
            )
        
        return ui.HTML(
            f'<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin-bottom:20px;">'
            f'<div style="font-size:24px;font-weight:700;color:{pred_color};margin-bottom:15px;">{pred_label}</div>'
            f'<div style="font-size:15px;margin-bottom:8px;">불량 확률: '
            f'<strong style="font-size:18px;color:#dc3545;">{prob:.4f}</strong> '
            f'(임계값: {model_manager.threshold:.4f})</div>{forced_msg}</div>'
        )
    
    @output
    @render.plot
    def shap_waterfall_plot():
        result = analysis_result.get()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if result is None:
            ax.axis("off")
            plt.tight_layout()
            return fig
        
        explanation = result.get('explanation')
        if explanation is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "SHAP 생성 실패", ha="center", va="center",
                   fontsize=14, color="#dc3545")
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
            ax.text(0.5, 0.5, "Plot 생성 실패", ha="center", va="center",
                   fontsize=12, color="#dc3545")
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
            
            item_html = (
                f'<li style="margin-bottom:14px;">'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<span style="background:#2A2D30;color:white;border-radius:50%;width:22px;height:22px;'
                f'display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">{rank}</span>'
                f'<strong style="font-size:14px;">{item["label"]}</strong></div>'
                f'<div style="margin-left:28px;margin-top:3px;">'
                f'<span style="color:#6c757d;font-size:12px;">현재 값: {val}</span><br>'
            )
            
            rec = recommendations.get(feat_name, {})
            if rec:
                if rec.get("type") == "numeric":
                    min_v = format_value(rec.get("min"))
                    max_v = format_value(rec.get("max"))
                    status = rec.get("status")
                    if status == "no-normal-but-best":
                        item_html += (
                            f'<span style="color:#856404;font-weight:600;font-size:12px;">'
                            f'• 정상 전환 불가, 확률 최소화 후보: {min_v} ~ {max_v}</span><br>'
                        )
                    else:
                        item_html += (
                            f'<span style="color:#28a745;font-weight:600;font-size:12px;">'
                            f'✓ 정상 전환 추천: {min_v} ~ {max_v}</span><br>'
                        )
                elif rec.get("type") == "categorical":
                    values = ", ".join(rec.get("values", []))
                    status = rec.get("status")
                    if status == "no-normal-but-best":
                        item_html += (
                            f'<span style="color:#856404;font-weight:600;font-size:12px;">'
                            f'• 정상 전환 불가, 확률 최소화 후보: {values}</span><br>'
                        )
                    else:
                        item_html += (
                            f'<span style="color:#28a745;font-weight:600;font-size:12px;">'
                            f'✓ 정상 전환 추천 값: {values}</span><br>'
                        )
            
            item_html += (
                f'<span style="font-size:12px;">SHAP 기여도: </span>'
                f'<span style="color:{color};font-weight:600;font-size:13px;">{contrib:+.4f}</span> '
                f'<span style="color:#6c757d;font-size:11px;">({direction})</span></div></li>'
            )
            items.append(item_html)
        
        best_prob = recommendations.get("best_probability")
        prob_html = ""
        if best_prob is not None:
            prob_html = (
                f'<div style="background:#d4edda;border-left:4px solid #28a745;padding:10px;'
                f'margin-top:14px;border-radius:4px;">'
                f'<strong style="font-size:13px;">📈 추천 적용 시 예상 불량 확률(최소): {best_prob:.4f}</strong> '
                f'<span style="color:#6c757d;font-size:11px;">(임계값: {model_manager.threshold:.4f})</span></div>'
            )
        
        html = (
            f'<div style="background:#f8f9fa;padding:18px;border-radius:8px;height:100%;">'
            f'<div style="font-weight:600;font-size:15px;margin-bottom:14px;border-bottom:2px solid #dee2e6;'
            f'padding-bottom:7px;">📊 불량 영향 상위 변수 및 정상/최소확률 전환 권고</div>'
            f'<ul style="padding-left:18px;margin:0;">{"".join(items)}</ul>{prob_html}</div>'
        )
        return ui.HTML(html)