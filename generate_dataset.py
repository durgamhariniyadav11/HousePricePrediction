# =============================================================================
# generate_dataset.py
# Downloads the "Real Estate Data from 7 Indian Cities" dataset from GitHub
# Source: github.com/aaryadevg/Real-Estate-ETL  (originally from Kaggle)
# Cities: Chennai, Mumbai, Delhi, Bangalore, Hyderabad, Kolkata, Pune
# Run this ONCE before train_model.py
# =============================================================================

import pandas as pd
import numpy as np
import re
import os

# ─────────────────────────────────────────────
# Raw CSV URL (GitHub — no login required)
# ─────────────────────────────────────────────
RAW_URL = (
    "https://raw.githubusercontent.com/aaryadevg/Real-Estate-ETL/"
    "main/data/raw/Real%20Estate%20Data%20V21.csv"
)

OUTPUT_FILE = "dataset.csv"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def parse_price(price_str: str) -> float:
    """
    Convert Indian price strings to numeric rupees.
    Examples: '₹1.99 Cr' → 19900000, '₹48.0 L' → 4800000
    """
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).replace("₹", "").replace(",", "").strip()
    try:
        if "Cr" in s:
            return float(s.replace("Cr", "").strip()) * 1_00_00_000
        elif "L" in s:
            return float(s.replace("L", "").strip()) * 1_00_000
        else:
            return float(s)
    except ValueError:
        return np.nan


def extract_city(location_str: str) -> str:
    """Extract city name from the Location column (last token after comma)."""
    if pd.isna(location_str):
        return "Unknown"
    parts = [p.strip() for p in str(location_str).split(",")]
    return parts[-1] if parts else "Unknown"


def extract_locality(location_str: str) -> str:
    """Extract locality (first token before comma)."""
    if pd.isna(location_str):
        return "Unknown"
    return str(location_str).split(",")[0].strip()


def extract_bedrooms(title_str: str) -> float:
    """Pull BHK count from property title, e.g. '3 BHK Flat...' → 3."""
    if pd.isna(title_str):
        return np.nan
    match = re.search(r'(\d+)\s*BHK', str(title_str), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def extract_property_type(title_str: str) -> str:
    """Classify property type from title."""
    if pd.isna(title_str):
        return "Apartment"
    t = str(title_str).lower()
    if "villa" in t:
        return "Villa"
    elif "independent house" in t or "independent/builder" in t:
        return "Independent House"
    elif "plot" in t or "land" in t:
        return "Plot"
    else:
        return "Apartment"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("[INFO] Downloading Indian Real Estate dataset (7 cities) from GitHub …")
    print(f"[INFO] Source: {RAW_URL}\n")

    try:
        df_raw = pd.read_csv(RAW_URL, low_memory=False)
        print(f"[INFO] Raw data loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        print(f"[INFO] Columns: {list(df_raw.columns)}\n")
    except Exception as e:
        print(f"[ERROR] Could not download dataset: {e}")
        print("[HINT] Check your internet connection and try again.")
        raise

    # ── Build clean dataframe ────────────────────────────────────────────────
    df = pd.DataFrame()

    df["price"]         = df_raw["Price"].apply(parse_price)
    df["price_per_sqft"]= pd.to_numeric(df_raw["Price_per_SQFT"], errors="coerce")
    df["total_area_sqft"]= pd.to_numeric(df_raw["Total_Area"],    errors="coerce")
    df["bathrooms"]     = pd.to_numeric(df_raw["Baths"],          errors="coerce")
    df["balcony"]       = df_raw["Balcony"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["bedrooms"]      = df_raw["Property Title"].apply(extract_bedrooms)
    df["property_type"] = df_raw["Property Title"].apply(extract_property_type)
    df["city"]          = df_raw["Location"].apply(extract_city)
    df["locality"]      = df_raw["Location"].apply(extract_locality)

    # ── Drop rows with missing target ────────────────────────────────────────
    before = len(df)
    df.dropna(subset=["price"], inplace=True)
    print(f"[INFO] Dropped {before - len(df)} rows with missing price.")

    # ── Fill missing numeric values ──────────────────────────────────────────
    df["bedrooms"].fillna(df["bedrooms"].median(), inplace=True)
    df["bathrooms"].fillna(df["bathrooms"].median(), inplace=True)
    df["total_area_sqft"].fillna(df["total_area_sqft"].median(), inplace=True)
    df["price_per_sqft"].fillna(df["price_per_sqft"].median(), inplace=True)

    # ── Remove extreme outliers (1st–99th percentile on price) ───────────────
    q_low  = df["price"].quantile(0.01)
    q_high = df["price"].quantile(0.99)
    df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]

    # ── Clip unrealistic values ──────────────────────────────────────────────
    df["bedrooms"]       = df["bedrooms"].clip(1, 10)
    df["bathrooms"]      = df["bathrooms"].clip(1, 10)
    df["total_area_sqft"]= df["total_area_sqft"].clip(200, 20000)
    df["price_per_sqft"] = df["price_per_sqft"].clip(500, 100000)

    # ── Normalise city names ─────────────────────────────────────────────────
    city_map = {
        "Chennai":   "Chennai",
        "Mumbai":    "Mumbai",
        "Delhi":     "Delhi",
        "Bangalore": "Bangalore",
        "Hyderabad": "Hyderabad",
        "Kolkata":   "Kolkata",
        "Pune":      "Pune",
    }
    df["city"] = df["city"].apply(
        lambda c: next((v for k, v in city_map.items() if k.lower() in str(c).lower()), "Other")
    )

    # ── Keep top localities per city ─────────────────────────────────────────
    top_localities = df["locality"].value_counts().nlargest(50).index
    df["locality"] = df["locality"].where(df["locality"].isin(top_localities), "Other")

    # ── Final column order ───────────────────────────────────────────────────
    df = df[[
        "city", "locality", "property_type",
        "bedrooms", "bathrooms", "balcony",
        "total_area_sqft", "price_per_sqft",
        "price"
    ]].reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DATASET SAVED → {OUTPUT_FILE}")
    print(f"{'='*55}")
    print(f"  Total records   : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Cities          : {sorted(df['city'].unique())}")
    print(f"  Price range     : ₹{df['price'].min():,.0f}  →  ₹{df['price'].max():,.0f}")
    print(f"  Avg price       : ₹{df['price'].mean():,.0f}")
    print(f"  Median price    : ₹{df['price'].median():,.0f}")
    print(f"\n  Records per city:")
    print(df["city"].value_counts().to_string())
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
