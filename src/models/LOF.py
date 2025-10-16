from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

try:
    import joblib
except ImportError:
    joblib = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1_time.csv"
TEST_FILE = BASE_DIR / "data" / "raw" / "test.csv"  # 이 부분도 일관성 있게

TARGET_COLUMN = "passorfail"
DROP_COLUMNS = (
    "id",
    "line",
    "name",
    "mold_name",
    "time",
    "date",
    "working",
    "emergency_stop",
    "registration_time",
    "tryshot_signal",
    "mold_code",
    "heating_furnace",
    TARGET_COLUMN,
)


df = pd.read_csv(DATA_FILE)

if TARGET_COLUMN in df.columns:
    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    normal_df = df.loc[labels == 0.0].copy()
    if normal_df.empty:
        normal_df = df.copy()
else:
    normal_df = df.copy()

columns_to_drop = [c for c in DROP_COLUMNS if c in normal_df.columns]

normal_df = normal_df.copy()
normal_df.loc[:, "mold_code"] = normal_df["mold_code"].fillna("").astype(str)
normal_df.loc[:, "mold_code"] = normal_df["mold_code"].replace({"": "UNKNOWN"})

contamination = 0.01
threshold_quantile = 0.01

models = {}
imputers = {}
feature_columns_map = {}
threshold_map = {}
train_summary = []

for mold_code in sorted(normal_df["mold_code"].unique()):
    subset = normal_df.loc[normal_df["mold_code"] == mold_code].copy()
    if subset.empty:
        continue

    subset_features = subset.drop(columns=columns_to_drop, errors="ignore")
    subset_features = subset_features.apply(pd.to_numeric, errors="coerce")
    subset_features = subset_features.dropna(axis=1, how="all")

    if subset_features.shape[1] == 0:
        print(f"Skipping mold_code={mold_code}: no numeric features.")
        continue

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(subset_features)
    if X_train.size == 0:
        print(f"Skipping mold_code={mold_code}: unable to build training matrix.")
        continue

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    model.fit(X_train)

    train_scores = model.decision_function(X_train)
    threshold = float(np.quantile(train_scores, threshold_quantile))

    models[mold_code] = model
    imputers[mold_code] = imputer
    feature_columns_map[mold_code] = subset_features.columns.tolist()
    threshold_map[mold_code] = threshold
    train_summary.append(
        {
            "mold_code": mold_code,
            "samples": len(subset_features),
            "features": subset_features.shape[1],
            "threshold": threshold,
            "train_anomalies": int((train_scores < threshold).sum()),
        }
    )

if not models:
    raise RuntimeError("No Isolation Forest models were trained; check training data.")

print("Isolation Forest fitted per mold_code.")
print(f"- Trained mold codes: {len(models)}")
for item in train_summary:
    print(
        "  · mold_code={mold_code}: samples={samples}, features={features}, "
        "threshold={threshold:.5f}, train_flags={train_anomalies}".format(**item)
    )

if joblib is not None:
    model_dir = BASE_DIR / "data" / "intermin" 
    model_dir.mkdir(parents=True, exist_ok=True)
    export_payload = {
        "models": models,
        "imputers": imputers,
        "feature_columns": feature_columns_map,
        "thresholds": threshold_map,
        "contamination": contamination,
        "threshold_quantile": threshold_quantile,
    }
    joblib.dump(export_payload, model_dir / "isolation_forest_by_mold_code.pkl")
    print(f"Saved models to {model_dir / 'isolation_forest_by_mold_code.pkl'}")
else:
    print("joblib not installed; skipping model export.")
test_df = pd.read_csv(TEST_FILE)
test_df = test_df.copy()
test_df.loc[:, "mold_code"] = test_df["mold_code"].fillna("").astype(str)
test_df.loc[:, "mold_code"] = test_df["mold_code"].replace({"": "UNKNOWN"})

print("Streaming predictions on test.csv:")
score_history = {code: [] for code in models}
index_history = {code: [] for code in models}
anomaly_history = {code: [] for code in models}

for idx, row in enumerate(test_df.itertuples(index=False), start=1):
    row_dict = row._asdict()
    code_value = row_dict.get("mold_code", "UNKNOWN")
    if pd.isna(code_value) or code_value == "":
        code_value = "UNKNOWN"
    else:
        code_value = str(code_value)

    if code_value not in models:
        print(f"row {idx}: mold_code={code_value} skipped (no trained model)")
        continue

    row_df = pd.DataFrame([row_dict])
    row_features = row_df.drop(columns=columns_to_drop, errors="ignore")
    row_features = row_features.apply(pd.to_numeric, errors="coerce")
    row_features = row_features.reindex(columns=feature_columns_map[code_value])
    row_array = imputers[code_value].transform(row_features)
    row_score = models[code_value].decision_function(row_array)[0]
    row_threshold = threshold_map[code_value]
    row_anomaly = row_score < row_threshold

    score_history[code_value].append(row_score)
    index_history[code_value].append(idx)
    anomaly_history[code_value].append(bool(row_anomaly))

    print(
        f"row {idx}: mold_code={code_value}, score={row_score:.5f}, anomaly={bool(row_anomaly)}"
    )

try:
    import matplotlib.pyplot as plt

    any_plotted = False
    for mold_code, scores in score_history.items():
        if not scores:
            continue

        indices = index_history[mold_code]
        flags = anomaly_history[mold_code]
        plt.figure(figsize=(10, 4))
        plt.plot(indices, scores, marker="o", label="decision score")
        plt.axhline(
            y=threshold_map[mold_code],
            color="red",
            linestyle="--",
            label="threshold",
        )
        flagged_indices = [i for i, flag in zip(indices, flags) if flag]
        flagged_scores = [s for s, flag in zip(scores, flags) if flag]
        if flagged_indices:
            plt.scatter(
                flagged_indices,
                flagged_scores,
                color="red",
                edgecolors="black",
                s=40,
                zorder=3,
                label="flagged anomalies",
            )
        plt.xlabel("Row index")
        plt.ylabel("Decision score")
        plt.title(f"Isolation Forest scores on test.csv (mold_code={mold_code})")
        plt.legend()
        plt.tight_layout()
        any_plotted = True

    if any_plotted:
        plt.show()
except Exception as exc:
    print(f"Visualization skipped: {exc}")
