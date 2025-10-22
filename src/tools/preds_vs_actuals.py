from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_FEATURE_FILE = BASE_DIR / "data" / "raw" / "test.csv"
TEST_LABEL_FILE = BASE_DIR / "data" / "raw" / "test_label.csv"
MODEL_FILE = BASE_DIR / "data" / "models" / "LightGBM_v3.pkl"
OUTPUT_FILE = BASE_DIR / "data" / "intermin" / "test_predictions_v3.csv"

# Load data -------------------------------------------------------------
features = pd.read_csv(TEST_FEATURE_FILE)
labels = pd.read_csv(TEST_LABEL_FILE)
test_df = features.merge(labels, on="id")
predict_df = test_df.copy()

# Load model ------------------------------------------------------------
model_artifact = joblib.load(MODEL_FILE)
model = model_artifact["model"]
scaler = model_artifact.get("scaler")
ordinal = model_artifact.get("ordinal_encoder")
onehot = model_artifact.get("onehot_encoder")

# Resolve feature columns ----------------------------------------------
numeric_cols = tuple(model_artifact.get("numeric_columns") or getattr(scaler, "feature_names_in_", ()))
cat_cols = tuple(model_artifact.get("categorical_columns") or getattr(ordinal, "feature_names_in_", ()))
resolved_numeric = [col for col in numeric_cols if col in test_df.columns]
resolved_cat = [col for col in cat_cols if col in test_df.columns]

# Numeric preprocessing -------------------------------------------------
numeric_matrix = np.empty((len(test_df), 0))
if resolved_numeric:
    if scaler is not None:
        numeric_matrix = scaler.transform(predict_df[resolved_numeric])
    else:
        numeric_matrix = predict_df[resolved_numeric].to_numpy(dtype=float)

# Categorical preprocessing --------------------------------------------
cat_matrix = np.empty((len(test_df), 0))
if resolved_cat:
    if ordinal is not None:
        encoded_columns = []
        unknown_value = getattr(ordinal, "unknown_value", -1)
        for idx, col in enumerate(resolved_cat):
            categories = ordinal.categories_[idx]
            series = predict_df[col]
            try:
                series = series.astype(categories.dtype)
            except (TypeError, ValueError):
                series = series.astype("string")
                categories = categories.astype("string")
            mapping = {cat: code for code, cat in enumerate(categories)}
            encoded = series.map(mapping).fillna(unknown_value).to_numpy()
            encoded_columns.append(encoded.reshape(-1, 1))
        cat_codes = np.hstack(encoded_columns) if encoded_columns else np.empty((len(test_df), 0))
    else:
        cat_codes = predict_df[resolved_cat].astype("category").apply(lambda col: col.cat.codes).to_numpy()
    if onehot is not None:
        cat_matrix = onehot.transform(cat_codes)
    else:
        cat_matrix = cat_codes

# Combine matrices ------------------------------------------------------
if numeric_matrix.size and cat_matrix.size:
    X_processed = np.hstack([numeric_matrix, cat_matrix])
elif numeric_matrix.size:
    X_processed = numeric_matrix
elif cat_matrix.size:
    X_processed = cat_matrix
else:
    X_processed = np.zeros((len(predict_df), 0))

# Inference -------------------------------------------------------------
threshold = float(model_artifact.get("operating_threshold", 0.5))
best_iter = getattr(model, "best_iteration", None)
if best_iter:
    probabilities = model.predict(X_processed, num_iteration=best_iter)
else:
    probabilities = model.predict(X_processed)
predictions = (probabilities >= threshold).astype(int)

# Attach to original dataframe -----------------------------------------
test_df["probability"] = probabilities
test_df["prediction"] = predictions

override_mask = test_df["tryshot_signal"] == "D"
test_df.loc[override_mask, "prediction"] = 1

# Metrics ---------------------------------------------------------------
actual = test_df["passorfail"].astype(int).to_numpy()
real_predictions = test_df["prediction"].astype(int).to_numpy()
print("성능 지표:")
print(f"  Accuracy : {accuracy_score(actual, real_predictions):.4f}")
print(f"  Precision: {precision_score(actual, real_predictions, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(actual, real_predictions, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(actual, real_predictions, zero_division=0):.4f}")
print()

print("예측 결과 미리보기:")
print(test_df[["id", "passorfail", "probability", "prediction"]].head())

# Save to CSV -----------------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
test_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n결과를 '{OUTPUT_FILE}' 경로에 저장했습니다.")
