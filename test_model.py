# =============================================================================
# test_model.py
# Loads the saved model + metadata and evaluates accuracy on the dataset.
# Prints MAE, RMSE, R², and MAPE metrics with a 80/20 split (same seed).
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH   = "model.pkl"
META_PATH    = "model_meta.pkl"
DATASET_PATH = "dataset.csv"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: '{MODEL_PATH}'. Run train_model.py first.")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found: '{META_PATH}'. Run train_model.py first.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: '{DATASET_PATH}'. Run generate_dataset.py first.")

    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    df    = pd.read_csv(DATASET_PATH)
    return model, meta, df


def prepare_features(df: pd.DataFrame, meta: dict):
    encoders     = meta["encoders"]
    feature_cols = meta["feature_cols"]

    df2 = df.copy()
    for col in ["city", "locality", "property_type"]:
        le = encoders.get(col)
        if le:
            df2[col] = df2[col].apply(
                lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
            )

    df2["bed_bath_ratio"]    = df2["bedrooms"]        / (df2["bathrooms"] + 1)
    df2["area_per_bedroom"]  = df2["total_area_sqft"] / (df2["bedrooms"]  + 1)
    df2["area_per_bathroom"] = df2["total_area_sqft"] / (df2["bathrooms"] + 1)

    X = df2[feature_cols]
    y = df2["price"]
    return X, y


def evaluate(model, X, y):
    # Same 80/20 split and random_state as train_model.py
    # Keep as DataFrame (not numpy array) so feature names are preserved
    idx_train, idx_test = train_test_split(
        X.index, test_size=0.2, random_state=42
    )
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    y_pred = model.predict(X_test)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    mape  = np.mean(np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, 1, y_test.values))) * 100

    # Also compute train R² to check for overfitting
    y_train_pred = model.predict(X_train)
    r2_train     = r2_score(y_train, y_train_pred)

    print("\n" + "=" * 60)
    print("           MODEL ACCURACY REPORT")
    print("=" * 60)
    print(f"  Test  samples          : {len(y_test):,}")
    print(f"  Train samples          : {len(y_train):,}")
    print("-" * 60)
    print(f"  MAE   (Mean Abs Error) : ₹{mae:>15,.0f}")
    print(f"  RMSE  (Root MSE)       : ₹{rmse:>15,.0f}")
    print(f"  MAPE  (Mean Abs % Err) : {mape:>14.2f} %")
    print(f"  R²    Score (Test)     : {r2:>14.4f}   ({r2*100:.2f}%)")
    print(f"  R²    Score (Train)    : {r2_train:>14.4f}   ({r2_train*100:.2f}%)")
    print("=" * 60)

    # Interpretation
    print("\n  Interpretation:")
    if r2 >= 0.90:
        grade = "Excellent"
    elif r2 >= 0.80:
        grade = "Good"
    elif r2 >= 0.65:
        grade = "Moderate"
    else:
        grade = "Needs improvement"

    print(f"  → R² = {r2:.4f}  →  {grade} fit")
    print(f"  → On average, predictions are off by ₹{mae/1e5:.2f} Lakhs")
    print(f"  → MAPE of {mape:.2f}% means predictions are within ~{mape:.1f}% of actual price")

    gap = r2_train - r2
    if gap > 0.10:
        print(f"  → Train R² ({r2_train:.4f}) vs Test R² ({r2:.4f}): gap={gap:.4f} — slight overfitting")
    else:
        print(f"  → Train/Test R² gap = {gap:.4f} — model generalises well")

    print("=" * 60 + "\n")

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "r2_train": r2_train, "mape": mape,
        "n_test": len(y_test), "n_train": len(y_train)
    }


if __name__ == "__main__":
    try:
        model, meta, df = load_artifacts()
        print(f"[INFO] Dataset loaded  : {len(df):,} rows")
        print(f"[INFO] Features        : {meta['feature_cols']}")
        X, y = prepare_features(df, meta)
        evaluate(model, X, y)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
