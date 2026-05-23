# =============================================================================
# train_model.py
# Cloud-Based House Price Prediction — Indian Real Estate (7 Cities)
# Trains RandomForestRegressor on the cleaned multi-city dataset
# Saves model + label encoders using joblib
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "dataset.csv"
MODEL_PATH   = "model.pkl"
META_PATH    = "model_meta.pkl"   # encoders + feature names


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the cleaned Indian housing CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'\n"
            "Run  python generate_dataset.py  first to download it."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[INFO] Cities: {sorted(df['city'].unique())}")
    return df


# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, engineer features, return X / y / metadata.
    """
    df = df.copy()

    # ── Encode categorical columns ───────────────────────────────────────────
    encoders = {}
    cat_cols = ["city", "locality", "property_type"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Encoded '{col}' — {len(le.classes_)} unique values")

    # ── Feature engineering ──────────────────────────────────────────────────
    df["bed_bath_ratio"]    = df["bedrooms"]        / (df["bathrooms"] + 1)
    df["area_per_bedroom"]  = df["total_area_sqft"] / (df["bedrooms"]  + 1)
    df["area_per_bathroom"] = df["total_area_sqft"] / (df["bathrooms"] + 1)

    # ── Feature list ─────────────────────────────────────────────────────────
    feature_cols = [
        "city", "locality", "property_type",
        "bedrooms", "bathrooms", "balcony",
        "total_area_sqft", "price_per_sqft",
        "bed_bath_ratio", "area_per_bedroom", "area_per_bathroom"
    ]

    X = df[feature_cols]
    y = df["price"]

    print(f"\n[INFO] Features  : {feature_cols}")
    print(f"[INFO] Samples   : {len(y):,}")
    return X, y, feature_cols, encoders


# ─────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    """80/20 split → train RandomForest → print metrics."""
    # Split on index so X_train/X_test stay as DataFrames (preserves feature names)
    idx_train, idx_test = train_test_split(
        X.index, test_size=0.2, random_state=42
    )
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    print(f"\n[INFO] Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Training RandomForestRegressor …")
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 52)
    print("         MODEL EVALUATION METRICS")
    print("=" * 52)
    print(f"  MAE  (Mean Absolute Error)  : ₹{mae:,.0f}")
    print(f"  RMSE (Root Mean Sq. Error)  : ₹{rmse:,.0f}")
    print(f"  R²   Score                  : {r2:.4f}")
    print("=" * 52 + "\n")

    return model, X_test, y_test


# ─────────────────────────────────────────────
# 4. Save Artifacts
# ─────────────────────────────────────────────
def save_artifacts(model, feature_cols: list, encoders: dict):
    """Persist model and encoder metadata for use in app.py."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols, "encoders": encoders}, META_PATH)
    print(f"[INFO] Model saved    → '{MODEL_PATH}'")
    print(f"[INFO] Metadata saved → '{META_PATH}'")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df                             = load_data(DATASET_PATH)
        X, y, feature_cols, encoders   = preprocess(df)
        model, X_test, y_test          = train_and_evaluate(X, y)
        save_artifacts(model, feature_cols, encoders)
        print("[SUCCESS] Run  streamlit run app.py  to launch the dashboard.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
