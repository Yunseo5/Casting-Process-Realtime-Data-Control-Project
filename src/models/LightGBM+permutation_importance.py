import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTENC
import joblib
import multiprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from statistics import median
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score

# ---------------------------
# 설정
# ---------------------------
RANDOM_STATE = 2025
N_TRIALS = 40
N_FOLDS = 5

CPU_COUNT = multiprocessing.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1)

# num_boost_round: early stopping을 not-supported 환경 대비 고정 부스트 반복 수 (필요 시 증가)
NUM_BOOST_ROUND = 200
PI_METRICS = ("f1", "roc_auc")
PI_N_REPEATS = 20

# ---------------------------
# 파일 경로
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v1_time.csv"        
DATA_FILE_TEST = BASE_DIR / "data" / "processed" / "valid_v1_time.csv"          
MODEL_OUTPUT = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"           
SCORE_OUTPUT = BASE_DIR / "data" / "models" / "LightGBM_v1_scores.csv"    

# ---------------------------
# 데이터 로드
# ---------------------------
train_df = pd.read_csv(DATA_FILE_TRAIN)
test_df = pd.read_csv(DATA_FILE_TEST)

train_df.drop(columns=["date", "time", "Unnamed: 0"], inplace=True, errors='ignore')
test_df.drop(columns=["date", "time", "Unnamed: 0"], inplace=True, errors='ignore')

target_col = "passorfail"

y_train = train_df[target_col].astype(int).copy()
X_train = train_df.drop(columns=[target_col]).copy()
feature_names = X_train.columns.tolist()

y_test = test_df[target_col].astype(int).copy()
X_test = test_df.drop(columns=[target_col]).copy()
X_test_original = X_test.copy()

# 수치형/범주형 컬럼 분리
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------------------------
# 유틸 함수
# pr_auc_score = PR-AUC
# find_best_threshold_for_f1 = f1기준 threshold 찾기
# ---------------------------
def pr_auc_score(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def find_best_threshold_for_f1(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_f1 = -1.0
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0

    # 모든 threshold에 대해 F1 계산하고 최대 F1인 threshold 선택
    for i, thr in enumerate(thresholds):
        prec_i = precision[i + 1]
        rec_i = recall[i + 1]
        if prec_i + rec_i <= 0:
            continue
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i)
        if f1_i > best_f1:
            best_f1 = f1_i
            best_thr = thr
            best_prec = prec_i
            best_rec = rec_i

    # thresholds가 없거나 best_f1을 찾지 못한 경우 0.5 기준으로 계산
    if best_f1 < 0:
        pred_bin = (y_scores >= 0.5).astype(int)
        return 0.5, f1_score(y_true, pred_bin, zero_division=0), precision_score(y_true, pred_bin, zero_division=0), recall_score(y_true, pred_bin, zero_division=0)

    return best_thr, best_f1, best_prec, best_rec


class LightGBMPermutationWrapper(BaseEstimator, ClassifierMixin):
    """학습된 LightGBM 부스터를 sklearn 퍼뮤테이션 중요도와 호환되게 감싸는 래퍼."""

    def __init__(
        self,
        booster,
        feature_names,
        numeric_cols,
        cat_cols,
        scaler,
        ordinal_encoder,
        onehot_encoder,
        operating_threshold,
        num_iteration=None,
    ):
        self.booster = booster
        self.feature_names = list(feature_names)
        self.numeric_cols = list(numeric_cols)
        self.cat_cols = list(cat_cols)
        self.scaler = scaler
        self.ordinal_encoder = ordinal_encoder
        self.onehot_encoder = onehot_encoder
        self.operating_threshold = float(operating_threshold)
        self.num_iteration = num_iteration
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = np.array([0, 1], dtype=int)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        return self

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names)
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"입력 데이터에 필요한 피처 {missing} 가(이) 없습니다.")
        return df

    def _transform(self, X_df):
        n_samples = len(X_df)

        if self.numeric_cols:
            X_num = X_df[self.numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            if self.scaler is not None:
                X_num = self.scaler.transform(X_num)
        else:
            X_num = np.empty((n_samples, 0), dtype=np.float64)

        if self.cat_cols:
            if self.ordinal_encoder is None or self.onehot_encoder is None:
                raise ValueError("범주형 인코더가 누락되었습니다.")
            X_cat = X_df[self.cat_cols].astype(str)
            X_cat_ord = self.ordinal_encoder.transform(X_cat)
            X_cat_ohe = self.onehot_encoder.transform(X_cat_ord.astype(int))
        else:
            X_cat_ohe = np.empty((n_samples, 0), dtype=np.float64)

        if X_num.size and X_cat_ohe.size:
            return np.hstack([X_num, X_cat_ohe])
        if X_num.size:
            return X_num
        if X_cat_ohe.size:
            return X_cat_ohe
        return np.zeros((n_samples, 0), dtype=np.float64)

    def predict_proba(self, X):
        X_df = self._ensure_dataframe(X)
        features = self._transform(X_df)
        proba_pos = self.booster.predict(features, num_iteration=self.num_iteration)
        proba_pos = np.asarray(proba_pos, dtype=np.float64)
        proba_pos = np.clip(proba_pos, 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        X_df = self._ensure_dataframe(X)
        proba = self.predict_proba(X_df)[:, 1]
        preds = (proba >= self.operating_threshold).astype(int)
        if "tryshot_signal" in X_df.columns:
            mask = X_df["tryshot_signal"].astype(str).str.upper() == "D"
            preds[mask.to_numpy(dtype=bool)] = 1
        return preds


def _resolve_metric(metric):
    metric_str = str(metric).lower()
    if metric_str in {"f1", "f1_score"}:
        return "f1", lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, zero_division=0)
    if metric_str in {"roc_auc", "roc-auc", "auc"}:
        return "roc_auc", lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba)
    if metric_str in {"average_precision", "avg_precision", "ap", "pr_auc", "pr-auc"}:
        return "average_precision", lambda y_true, y_pred, y_proba: average_precision_score(y_true, y_proba)
    raise ValueError(f"지원하지 않는 metric: {metric}. 지원 목록: f1, roc_auc, average_precision")


def compute_permutation_importance_table(estimator, X, y, metrics, n_repeats, random_state, n_jobs):
    if n_jobs not in (-1, 1):
        print("경고: 현재 구현은 n_jobs 옵션을 지원하지 않아 단일 프로세스로 동작합니다.", file=sys.stderr)

    X_df = X.copy()
    rng = np.random.default_rng(seed=random_state)

    metric_funcs = []
    base_scores = {}
    y_pred_base = estimator.predict(X_df)
    y_proba_base = estimator.predict_proba(X_df)[:, 1]

    for metric in metrics:
        metric_name, metric_fn = _resolve_metric(metric)
        metric_funcs.append((metric_name, metric_fn))
        base_scores[metric_name] = metric_fn(y, y_pred_base, y_proba_base)

    records = []
    feature_values = {col: X_df[col].to_numpy(copy=True) for col in estimator.feature_names}

    for feature in estimator.feature_names:
        drops = {metric_name: [] for metric_name, _ in metric_funcs}
        original_values = feature_values[feature]

        for _ in range(n_repeats):
            permuted = original_values.copy()
            rng.shuffle(permuted)
            X_perm = X_df.copy()
            X_perm[feature] = permuted

            y_pred_perm = estimator.predict(X_perm)
            y_proba_perm = estimator.predict_proba(X_perm)[:, 1]

            for metric_name, metric_fn in metric_funcs:
                perm_score = metric_fn(y, y_pred_perm, y_proba_perm)
                drops[metric_name].append(base_scores[metric_name] - perm_score)

        for metric_name, drop_values in drops.items():
            drop_array = np.array(drop_values, dtype=np.float64)
            records.append(
                {
                    "metric": metric_name,
                    "feature": feature,
                    "importance_mean": float(drop_array.mean()),
                    "importance_std": float(drop_array.std(ddof=1)) if len(drop_array) > 1 else 0.0,
                }
            )

    fi_df = pd.DataFrame.from_records(records)
    fi_df = fi_df.sort_values(["metric", "importance_mean"], ascending=[True, False]).reset_index(drop=True)
    metric_labels = [metric_name for metric_name, _ in metric_funcs]
    return fi_df, metric_labels


def plot_permutation_importance_table(fi_df, metrics, output_path, top_k=None):
    metrics = list(dict.fromkeys(metrics))
    if top_k is not None and top_k <= 0:
        top_k = None
    fig_rows = len(metrics)
    fig_height = max(3.0, (top_k or len(fi_df["feature"].unique())) * 0.4)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(10, fig_height * fig_rows), squeeze=False)

    for row_idx, metric in enumerate(metrics):
        ax = axes[row_idx, 0]
        subset = fi_df[fi_df["metric"] == metric]
        if subset.empty:
            ax.set_title(f"{metric} (데이터 없음)")
            ax.axis("off")
            continue
        subset = subset.sort_values("importance_mean")
        if top_k:
            subset = subset.tail(top_k)
        ax.barh(
            subset["feature"],
            subset["importance_mean"],
            xerr=subset["importance_std"],
            color="#1f77b4",
            alpha=0.9,
        )
        ax.set_title(f"{metric} 퍼뮤테이션 중요도")
        ax.set_xlabel("평균 점수 감소량")
        ax.set_ylabel("피처")
        ax.axvline(0.0, color="#555555", linewidth=0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Optuna objective
# ---------------------------
def objective(trial):
    # LightGBM 하이퍼파라미터
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "metric": "auc",
        "num_threads": N_JOBS,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    fold_f1s = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        # Ordinal encoding (SMOTENC 적용을 위해)
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_cat_ord = ord_enc.fit_transform(X_tr[cat_cols]).astype(int)
        X_val_cat_ord = ord_enc.transform(X_val[cat_cols]).astype(int)
        
        # Standard scaling
        scaler = StandardScaler()
        X_tr_num_scaled = scaler.fit_transform(X_tr[numeric_cols])
        X_val_num_scaled = scaler.transform(X_val[numeric_cols])
        
        X_tr_for_smote = np.hstack([X_tr_num_scaled, X_tr_cat_ord]) if (X_tr_num_scaled.size or X_tr_cat_ord.size) else np.empty((len(X_tr), 0))

        cat_feature_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
        sm = SMOTENC(categorical_features=cat_feature_indices, random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_tr_for_smote, y_tr.values)
        
        X_res_num = X_res[:, :len(numeric_cols)]
        
        X_res_cat_ord = X_res[:, len(numeric_cols):].astype(int)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # Smote 이전의 학습 데이터 기준으로 fit
        ohe.fit(X_tr_cat_ord)
        X_res_cat_ohe = ohe.transform(X_res_cat_ord)
        X_val_cat_ohe = ohe.transform(X_val_cat_ord)
        
        X_res_final = np.hstack([X_res_num, X_res_cat_ohe]) if (X_res_num.size or X_res_cat_ohe.size) else np.zeros((len(y_res), 0))
        X_val_final = np.hstack([X_val_num_scaled, X_val_cat_ohe]) if (X_val_num_scaled.size or X_val_cat_ohe.size) else np.zeros((len(X_val), 0))

        # 학습
        train_set = lgb.Dataset(X_res_final, label=y_res, free_raw_data=True)
        bst = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_set],
        )

        # 검증
        val_probs = bst.predict(X_val_final, num_iteration=NUM_BOOST_ROUND)

        # F1을 기준으로 최적 임계값 탐색
        thr, f1_at_thr, prec_at_thr, rec_at_thr = find_best_threshold_for_f1(y_val, val_probs)
        # 폴드별 F1 수집
        fold_f1s.append(f1_at_thr)

    # 폴드별 평균 F1 반환
    return float(np.mean(fold_f1s))

# ---------------------------
# Optuna study 실행
# ---------------------------
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgb_f1")
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

# ---------------------------
# CV 기반: 폴드별(원본 validation) threshold 수집 -> median을 임계값으로 사용
# (이제 각 폴드에서 F1을 최대화하는 임계값을 수집)
# ---------------------------
best_params = study.best_trial.params

final_lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": RANDOM_STATE,
    "metric": "auc",
    "num_threads": N_JOBS,
    "learning_rate": best_params["learning_rate"],
    "num_leaves": int(best_params["num_leaves"]),
    "max_depth": int(best_params["max_depth"]),
    "min_child_samples": int(best_params["min_child_samples"]),
    "subsample": float(best_params["subsample"]),
    "colsample_bytree": float(best_params["colsample_bytree"]),
    "reg_alpha": float(best_params["reg_alpha"]),
    "reg_lambda": float(best_params["reg_lambda"]),
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_thresholds = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr = X_train.iloc[tr_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_tr = y_train.iloc[tr_idx].copy()
    y_val = y_train.iloc[val_idx].copy()

    ord_enc_cv = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_cat_ord = ord_enc_cv.fit_transform(X_tr[cat_cols]).astype(int)
    X_val_cat_ord = ord_enc_cv.transform(X_val[cat_cols]).astype(int)

    scaler_cv = StandardScaler()
    X_tr_num_scaled = scaler_cv.fit_transform(X_tr[numeric_cols])
    X_val_num_scaled = scaler_cv.transform(X_val[numeric_cols])
    
    X_tr_for_smote = np.hstack([X_tr_num_scaled, X_tr_cat_ord]) if (X_tr_num_scaled.size or X_tr_cat_ord.size) else np.empty((len(X_tr), 0))
    
    cat_feature_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
    sm_cv = SMOTENC(categorical_features=cat_feature_indices, random_state=RANDOM_STATE)
    X_res_cv, y_res_cv = sm_cv.fit_resample(X_tr_for_smote, y_tr.values)

    X_res_num_cv = X_res_cv[:, :len(numeric_cols)]
    
    X_res_cat_ord_cv = X_res_cv[:, len(numeric_cols):].astype(int)
    ohe_cv = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_cv.fit(X_tr_cat_ord)
    X_res_cat_ohe_cv = ohe_cv.transform(X_res_cat_ord_cv)
    X_val_cat_ohe_cv = ohe_cv.transform(X_val_cat_ord)
    
    X_res_final_cv = np.hstack([X_res_num_cv, X_res_cat_ohe_cv]) if (X_res_num_cv.size or X_res_cat_ohe_cv.size) else np.zeros((len(y_res_cv), 0))
    X_val_final_cv = np.hstack([X_val_num_scaled, X_val_cat_ohe_cv]) if (X_val_num_scaled.size or X_val_cat_ohe_cv.size) else np.zeros((len(X_val), 0))

    dtrain_cv = lgb.Dataset(X_res_final_cv, label=y_res_cv, free_raw_data=True)
    bst_cv = lgb.train(final_lgb_params, dtrain_cv, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dtrain_cv])

    val_probs_cv = bst_cv.predict(X_val_final_cv, num_iteration=NUM_BOOST_ROUND)
    thr_cv, f1_cv, prec_cv, rec_cv = find_best_threshold_for_f1(y_val, val_probs_cv)

    fold_thresholds.append(thr_cv)
    fold_precisions.append(prec_cv)
    fold_recalls.append(rec_cv)
    fold_f1s.append(f1_cv)

# f1 기준 최적 임계값의 중앙값을 임계값으로 사용
thr_oper = float(median(fold_thresholds))
print(f"Selected operating threshold (median of folds): {thr_oper:.6f}")

# ---------------------------
# 전처리
# ---------------------------
ord_enc_full = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_cat_ord_full = ord_enc_full.fit_transform(X_train[cat_cols]).astype(int)
X_test_cat_ord_full = ord_enc_full.transform(X_test[cat_cols]).astype(int)

scaler_full = StandardScaler()
X_train_num_scaled_full = scaler_full.fit_transform(X_train[numeric_cols])
X_test_num_scaled_full = scaler_full.transform(X_test[numeric_cols])

X_train_for_smote_full = np.hstack([X_train_num_scaled_full, X_train_cat_ord_full]) if (X_train_num_scaled_full.size or X_train_cat_ord_full.size) else np.empty((len(X_train), 0))

cat_feature_indices_full = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
sm_full = SMOTENC(categorical_features=cat_feature_indices_full, random_state=RANDOM_STATE)
X_res_full, y_res_full = sm_full.fit_resample(X_train_for_smote_full, y_train.values)

X_res_num_full = X_res_full[:, :len(numeric_cols)]

X_res_cat_ord_full = X_res_full[:, len(numeric_cols):].astype(int)
ohe_full = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_full.fit(X_train_cat_ord_full)
X_res_cat_ohe_full = ohe_full.transform(X_res_cat_ord_full)
X_test_cat_ohe_full = ohe_full.transform(X_test_cat_ord_full)

X_res_final_full = np.hstack([X_res_num_full, X_res_cat_ohe_full]) if (X_res_num_full.size or X_res_cat_ohe_full.size) else np.zeros((len(y_res_full), 0))
X_test_final = np.hstack([X_test_num_scaled_full, X_test_cat_ohe_full]) if (X_test_num_scaled_full.size or X_test_cat_ohe_full.size) else np.zeros((len(X_test), 0))

# ---------------------------
# 최종 LightGBM 학습
# ---------------------------
train_set_full = lgb.Dataset(X_res_final_full, label=y_res_full, free_raw_data=True)
bst_final = lgb.train(final_lgb_params, train_set_full, num_boost_round=NUM_BOOST_ROUND, valid_sets=[train_set_full])

# ---------------------------
# 테스트 예측
# ---------------------------
y_test_proba = bst_final.predict(X_test_final, num_iteration=NUM_BOOST_ROUND)

y_test_pred_oper = (y_test_proba >= thr_oper).astype(int)

X_test["pred"] = y_test_pred_oper
X_test.loc[X_test["tryshot_signal"] == "D", "pred"] = 1  # tryshot_signal이 D인 경우 무조건 불량
y_test_pred_oper = X_test["pred"].values

# ---------------------------
# 테스트 평가
# ---------------------------
test_pr_auc = pr_auc_score(y_test, y_test_proba)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) == 2 else np.nan
test_acc = accuracy_score(y_test, y_test_pred_oper)
test_prec = precision_score(y_test, y_test_pred_oper, zero_division=0)
test_rec = recall_score(y_test, y_test_pred_oper, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred_oper, zero_division=0)
cm = confusion_matrix(y_test, y_test_pred_oper)

print("\n== Test set performance at operating threshold ==")
print(f"Operating threshold: {thr_oper:.6f}")
print(f"PR-AUC: {test_pr_auc:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
print("Confusion matrix:\n", cm)

# ---------------------------
# Permutation Importance
# ---------------------------
print("\n== Permutation importance (validation set) ==")

pi_wrapper = LightGBMPermutationWrapper(
    booster=bst_final,
    feature_names=feature_names,
    numeric_cols=numeric_cols,
    cat_cols=cat_cols,
    scaler=(scaler_full if numeric_cols else None),
    ordinal_encoder=(ord_enc_full if cat_cols else None),
    onehot_encoder=(ohe_full if cat_cols else None),
    operating_threshold=thr_oper,
    num_iteration=bst_final.best_iteration if bst_final.best_iteration else bst_final.current_iteration(),
)

pi_df, metric_labels = compute_permutation_importance_table(
    estimator=pi_wrapper,
    X=X_test_original,
    y=y_test.to_numpy(),
    metrics=PI_METRICS,
    n_repeats=PI_N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)

print(pi_df.head(20))

reports_dir = BASE_DIR / "reports"
pi_table_path = reports_dir / "permutation_importance_valid.csv"
pi_plot_path = reports_dir / "permutation_importance_valid.png"

reports_dir.mkdir(parents=True, exist_ok=True)
pi_df.to_csv(pi_table_path, index=False)
plot_permutation_importance_table(
    fi_df=pi_df,
    metrics=metric_labels,
    output_path=pi_plot_path,
    top_k=20,
)

print(f"\nPermutation importance table saved to: {pi_table_path}")
print(f"Permutation importance plot saved to: {pi_plot_path}")

