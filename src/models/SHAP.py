from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import joblib
import numpy as np
import pandas as pd
import shap

# ---------------------------
# 파일 경로
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v1_time.csv"
DATA_FILE_TEST = BASE_DIR / "data" / "raw" / "test.csv"
MODEL_OUTPUT = BASE_DIR / "data" / "models" / "LightGBM_v1.pkl"

# ---------------------------
# 데이터 로드
# ---------------------------
train_df = pd.read_csv(DATA_FILE_TRAIN)
test_df = pd.read_csv(DATA_FILE_TEST)

# LightGBM 학습 시 제거한 열과 타깃 처리
cols_to_drop = ["date", "time", "Unnamed: 0"]
target_col = "passorfail"

train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# ---------------------------
# 전처리 정보 준비
# ---------------------------
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# 저장된 아티팩트 로드
artifact = joblib.load(MODEL_OUTPUT)
bst = artifact["model"]
scaler = artifact.get("scaler")
ord_encoder = artifact.get("ordinal_encoder")
ohe_encoder = artifact.get("onehot_encoder")
oper_threshold = artifact.get("operating_threshold", 0.5)

# 숫자/범주 컬럼 별 전처리
if numeric_cols:
    X_test_num = scaler.transform(X_test[numeric_cols])
else:
    X_test_num = np.empty((len(X_test), 0))

if cat_cols:
    X_test_cat_ord = ord_encoder.transform(X_test[cat_cols]).astype(int)
    X_test_cat_ohe = ohe_encoder.transform(X_test_cat_ord)
else:
    X_test_cat_ohe = np.empty((len(X_test), 0))

X_test_final = (
    np.hstack([X_test_num, X_test_cat_ohe])
    if (X_test_num.size or X_test_cat_ohe.size)
    else np.zeros((len(X_test), 0))
)

# ---------------------------
# 예측 수행
# ---------------------------
test_proba = bst.predict(X_test_final)
test_pred = (test_proba >= oper_threshold).astype(int)

# tryshot_signal 규칙 반영 (LightGBM.py와 동일)
if "tryshot_signal" in test_df.columns:
    mask_tryshot_d = test_df["tryshot_signal"] == "D"
    test_pred[mask_tryshot_d.values] = 1

# ---------------------------
# SHAP 분석
# ---------------------------
feature_names = numeric_cols.copy()
if cat_cols:
    feature_names.extend(ohe_encoder.get_feature_names_out(cat_cols).tolist())
feature_names = np.array(feature_names, dtype=object)

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test_final)

if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

abs_shap = np.abs(shap_values)
top_feature_indices = abs_shap.argmax(axis=1)
top_feature_names = feature_names[top_feature_indices]

# ---------------------------
# 결과 열 추가
# ---------------------------
test_df["pred"] = test_pred
test_df["top_shap_feature"] = top_feature_names

# 필요 시 확인용 출력
print(test_df[["pred", "top_shap_feature"]].head())
test_df['pred'].value_counts()
test_df['top_shap_feature'].value_counts()
test_df[test_df['pred']==1]['top_shap_feature'].value_counts()
test_df[test_df['pred']==0]['top_shap_feature'].value_counts()

test_df['pred'].value_counts()

# ---------------------------
# SHAP 시각화 (직관형)
# ---------------------------
# 한글 폰트 설정 (Malgun Gothic, AppleGothic 우선 탐색)
font_candidates = ["Malgun Gothic", "AppleGothic"]
for font_name in font_candidates:
    if any(font.name == font_name for font in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font_name
        break
# 음수 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

mean_abs_shap = (
    pd.Series(abs_shap.mean(axis=0), index=feature_names)
    .sort_values()
)
top_k_bar = min(15, len(mean_abs_shap))
top_features = mean_abs_shap.tail(top_k_bar)

fig, ax = plt.subplots(figsize=(8, max(4, top_k_bar * 0.4)))
ax.barh(top_features.index, top_features.values, color="#1f77b4")
ax.set_xlabel("평균 절대 SHAP 값")
ax.set_title("LightGBM 예측에 영향력이 큰 피처 (상위)")
plt.tight_layout()
plt.show()

scatter_count = min(8, len(mean_abs_shap))
scatter_features = (
    mean_abs_shap.sort_values(ascending=False)
    .head(scatter_count)
    .index.tolist()
)

ok_mask = test_pred == 0
ng_mask = test_pred == 1

for feat_name in scatter_features:
    feat_indices = np.where(feature_names == feat_name)[0]
    if not len(feat_indices):
        continue
    feat_idx = int(feat_indices[0])
    shap_feat = shap_values[:, feat_idx]

    if feat_name in X_test.columns:
        raw_values = X_test[feat_name].values
        x_label = f"{feat_name} 원본 값"
    else:
        raw_values = X_test_final[:, feat_idx]
        x_label = f"{feat_name} (전처리 값)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        raw_values[ok_mask],
        shap_feat[ok_mask],
        color="#1f77b4",
        alpha=0.6,
        label="예측: 정상",
    )
    ax.scatter(
        raw_values[ng_mask],
        shap_feat[ng_mask],
        color="#d62728",
        alpha=0.7,
        label="예측: 불량",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"SHAP 값 ({feat_name})")
    ax.set_title(f"{feat_name} 값과 예측 기여도")
    ax.legend()
    plt.tight_layout()
    plt.show()
