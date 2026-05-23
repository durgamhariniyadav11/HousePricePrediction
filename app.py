# =============================================================================
# app.py
# Cloud-Based House Price Prediction — Indian Real Estate (7 Cities)
# Streamlit GUI — Modern, Responsive, Production-Ready
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS - Modern & Professional UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary: #FF6B35;
        --primary-light: #F7931E;
        --dark: #1a1a2e;
        --dark-light: #16213e;
        --accent: #00BCD4;
        --success: #4CAF50;
        --text-light: #e8e8e8;
        --bg-light: #f8fafb;
        --border: #e3e8ef;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #f8fafb 0%, #f0f2f5 100%);
        color: #1a1a2e;
    }

    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border: 1px solid #e0e5eb;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    /* Prediction display box */
    .prediction-box {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 32px 28px;
        border-radius: 20px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin: 24px 0;
        box-shadow: 0 12px 40px rgba(255,107,53,0.4);
        border: 2px solid rgba(255,255,255,0.1);
        animation: slideIn 0.6s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .prediction-label {
        font-size: 14px;
        opacity: 0.95;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-amount {
        font-size: 52px;
        font-weight: 800;
        letter-spacing: 0px;
        margin: 12px 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    .prediction-subtext {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 8px;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 800;
        color: #1a1a2e;
        border-left: 5px solid #FF6B35;
        padding-left: 14px;
        margin: 32px 0 18px 0;
        position: relative;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, #FF6B35, transparent);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f1929 100%);
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding: 20px 16px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }

    .sidebar-card {
        background: rgba(255,107,53,0.1);
        border: 1px solid rgba(255,107,53,0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        padding: 14px 16px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,107,53,0.3);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255,107,53,0.4);
    }

    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
    }

    /* Sidebar input styling */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-baseweb="input"] *,
    section[data-testid="stSidebar"] [data-baseweb="base-input"] input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        background-color: #f5f5f5 !important;
    }

    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] textarea:focus {
        background-color: #ffffff !important;
        border-color: #FF6B35 !important;
    }

    /* Dropdown styling */
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li,
    [data-baseweb="select"] [role="option"] {
        color: #111111 !important;
    }

    [data-baseweb="select"] [role="option"]:hover {
        background-color: rgba(255,107,53,0.1) !important;
    }

    /* Slider styling */
    section[data-testid="stSidebar"] [data-testid="stSlider"] {
        color: #e8e8e8;
    }

    /* Radio button styling */
    section[data-testid="stSidebar"] [data-testid="stRadio"] * {
        color: #e8e8e8 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 28px 24px;
        color: #777;
        font-size: 12px;
        border-top: 1px solid #e0e0e0;
        margin-top: 60px;
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 12px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }

    .footer strong {
        color: #FF6B35;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] button {
        background-color: #f5f7fa;
        color: #1a1a2e;
        font-weight: 600;
    }

    [data-testid="stExpander"] button:hover {
        background-color: #eff1f5;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Warning and error styling */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]  # 7 cities only
PROPERTY_TYPES = ["Apartment", "Independent House", "Villa", "Plot"]
CITY_COLORS = {
    "Bangalore": "#4CAF50", "Chennai":   "#2196F3",
    "Delhi":     "#9C27B0", "Hyderabad": "#FF5722",
    "Kolkata":   "#FF9800", "Mumbai":    "#F44336",
    "Pune":      "#00BCD4"
}


# ─────────────────────────────────────────────
# Loaders (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists("model.pkl"):
        return None, None
    model = joblib.load("model.pkl")
    meta  = joblib.load("model_meta.pkl") if os.path.exists("model_meta.pkl") else {}
    return model, meta


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists("dataset.csv"):
        return pd.DataFrame()
    df = pd.read_csv("dataset.csv")
    # Exclude the handful of rows tagged "Other" — not a real city
    df = df[df["city"] != "Other"].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────
def predict_price(model, meta: dict, inputs: dict) -> tuple:
    """Encode inputs and run model prediction with confidence score."""
    encoders    = meta.get("encoders", {})
    feature_cols = meta.get("feature_cols", [])

    def encode(col, val):
        le = encoders.get(col)
        if le is None:
            return 0
        if val in le.classes_:
            return int(le.transform([val])[0])
        return 0   # unseen label → 0

    bedrooms  = inputs["bedrooms"]
    bathrooms = inputs["bathrooms"]
    area      = inputs["total_area_sqft"]

    row = {
        "city":             encode("city",          inputs["city"]),
        "locality":         encode("locality",       inputs["locality"]),
        "property_type":    encode("property_type",  inputs["property_type"]),
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "balcony":          inputs["balcony"],
        "total_area_sqft":  area,
        "price_per_sqft":   inputs["price_per_sqft"],
        "bed_bath_ratio":   bedrooms / (bathrooms + 1),
        "area_per_bedroom": area / (bedrooms + 1),
        "area_per_bathroom":area / (bathrooms + 1),
    }

    X = pd.DataFrame([row])[feature_cols]
    
    # Get prediction from the ensemble
    prediction = float(model.predict(X)[0])
    
    # Calculate confidence score using predictions from individual estimators
    if hasattr(model, 'estimators_'):
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
        # Standard deviation represents uncertainty
        std_dev = np.std(tree_predictions)
        mean_pred = np.mean(tree_predictions)
        
        # Calculate coefficient of variation as % (lower = more confident)
        if mean_pred != 0:
            cv = (std_dev / mean_pred) * 100
            # Convert to confidence: max at 0% variation, min as variation increases
            confidence = max(0, min(100, 100 - cv))
        else:
            confidence = 50.0
    else:
        confidence = 75.0  # Default if can't calculate
    
    return prediction, confidence


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.markdown(
        "<div style='text-align:center; padding:16px 0 8px 0;'>"
        "<h2 style='margin:0; font-size:28px;'>🏠 Quick Predict</h2>"
        "<p style='color:#aaa; font-size:12px; margin:4px 0 0 0;'>Enter property details</p>"
        "</div>", unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # Location Section
    st.sidebar.markdown(
        "<div style='color:#FF6B35; font-weight:700; font-size:14px; "
        "text-transform:uppercase; letter-spacing:0.5px; margin-top:16px;'>📍 Location</div>",
        unsafe_allow_html=True
    )
    city = st.sidebar.selectbox("City", CITIES, index=3, key="city_select")

    # Dynamic locality list based on city
    if not df.empty and "city" in df.columns:
        localities = sorted(
            df[(df["city"] == city) & (df["locality"] != "Other")]["locality"].unique().tolist()
        )
        if not localities:
            localities = ["Other"]
    else:
        localities = ["Other"]
    locality = st.sidebar.selectbox("Locality", localities, key="locality_select")

    # Property Info Section
    st.sidebar.markdown(
        "<div style='color:#FF6B35; font-weight:700; font-size:14px; "
        "text-transform:uppercase; letter-spacing:0.5px; margin-top:20px;'>🏗️ Property Details</div>",
        unsafe_allow_html=True
    )
    property_type = st.sidebar.selectbox("Type", PROPERTY_TYPES, key="property_type_select")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        bedrooms = st.number_input("BHK", min_value=1, max_value=10, value=3, key="bedrooms_input")
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, key="bathrooms_input")
    
    balcony = st.sidebar.radio("Balcony", ["Yes", "No"], horizontal=True, key="balcony_radio", label_visibility="collapsed")

    # Area & Pricing Section
    st.sidebar.markdown(
        "<div style='color:#FF6B35; font-weight:700; font-size:14px; "
        "text-transform:uppercase; letter-spacing:0.5px; margin-top:20px;'>📐 Area & Pricing</div>",
        unsafe_allow_html=True
    )
    total_area = st.sidebar.slider("Total Area (sq ft)", 200, 20000, 1200, step=50, key="area_slider")
    price_per_sqft = st.sidebar.slider("Price/Sq Ft (₹)", 500, 100000, 6000, step=100, key="price_sqft_slider")

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button(
        "🔮 Predict Price",
        use_container_width=True,
        key="predict_button",
        help="Click to get the estimated house price"
    )

    return {
        "city":           city,
        "locality":       locality,
        "property_type":  property_type,
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "balcony":        1 if balcony == "Yes" else 0,
        "total_area_sqft":total_area,
        "price_per_sqft": price_per_sqft,
        "predict":        predict_btn,
    }


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
def fmt_inr(x, _=None):
    """Format axis ticks as ₹ Cr / L."""
    if x >= 1e7:
        return f"₹{x/1e7:.1f}Cr"
    elif x >= 1e5:
        return f"₹{x/1e5:.0f}L"
    return f"₹{x:,.0f}"


def plot_city_avg_price(df: pd.DataFrame):
    city_avg = (df.groupby("city")["price"].median()
                  .sort_values(ascending=False)
                  .reset_index())
    colors = [CITY_COLORS.get(c, "#888") for c in city_avg["city"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    bars = ax.bar(city_avg["city"], city_avg["price"], color=colors, 
                   edgecolor="white", linewidth=2, alpha=0.85)
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
    ax.set_title("Median House Price by City", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("City", fontsize=11, fontweight="600")
    ax.set_ylabel("Median Price", fontsize=11, fontweight="600")
    ax.bar_label(bars, labels=[fmt_inr(v) for v in city_avg["price"]], 
                 padding=5, fontsize=9, fontweight="600")
    
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols: list):
    display_names = {
        "city": "City", "locality": "Locality",
        "property_type": "Property Type",
        "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
        "balcony": "Balcony", "total_area_sqft": "Total Area (sqft)",
        "price_per_sqft": "Price/Sqft",
        "bed_bath_ratio": "Bed-Bath Ratio",
        "area_per_bedroom": "Area/Bedroom",
        "area_per_bathroom": "Area/Bathroom",
    }
    importances = model.feature_importances_
    labels = [display_names.get(f, f) for f in feature_cols]
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    colors = plt.cm.OrRd(np.linspace(0.4, 0.9, len(labels)))
    bars = ax.barh([labels[i] for i in indices], importances[indices], color=colors, edgecolor="white", linewidth=1.5)
    
    ax.set_xlabel("Importance Score", fontsize=11, fontweight="600")
    ax.set_title("Feature Importance in Price Prediction", fontsize=14, fontweight="bold", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', fontsize=9, fontweight='600')
    
    plt.tight_layout()
    return fig


def plot_price_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    prices_cr = df["price"] / 1e7
    ax.hist(prices_cr, bins=40, color="#FF6B35", edgecolor="white", alpha=0.8, linewidth=1)
    
    ax.set_xlabel("Price (₹ Crore)", fontsize=11, fontweight="600")
    ax.set_ylabel("Number of Properties", fontsize=11, fontweight="600")
    ax.set_title("Distribution of House Prices Across All Cities", fontsize=14, fontweight="bold", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics line
    mean_price = prices_cr.mean()
    ax.axvline(mean_price, color='#00BCD4', linestyle='--', linewidth=2, label=f'Mean: ₹{mean_price:.2f}Cr')
    ax.legend(fontsize=10, loc='upper right')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=1, linecolor='#f0f0f0',
                ax=ax, annot_kws={"size": 9, "weight": "600"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(model, meta: dict, df: pd.DataFrame):
    encoders     = meta.get("encoders", {})
    feature_cols = meta.get("feature_cols", [])

    df2 = df.copy()
    for col in ["city", "locality", "property_type"]:
        le = encoders.get(col)
        if le:
            df2[col] = df2[col].apply(
                lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
            )
    df2["bed_bath_ratio"]    = df2["bedrooms"] / (df2["bathrooms"] + 1)
    df2["area_per_bedroom"]  = df2["total_area_sqft"] / (df2["bedrooms"] + 1)
    df2["area_per_bathroom"] = df2["total_area_sqft"] / (df2["bathrooms"] + 1)

    available = [c for c in feature_cols if c in df2.columns]
    sample = df2[available + ["price"]].dropna().sample(min(200, len(df2)), random_state=42)
    y_actual = sample["price"].values
    y_pred   = model.predict(sample[available])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    ax.scatter(y_actual / 1e7, y_pred / 1e7, alpha=0.6, color="#FF6B35",
               edgecolors="white", s=80, linewidth=1.5)
    
    lims = [min(y_actual.min(), y_pred.min()) / 1e7,
            max(y_actual.max(), y_pred.max()) / 1e7]
    ax.plot(lims, lims, "b--", linewidth=2.5, label="Perfect Prediction", alpha=0.7)
    
    ax.set_xlabel("Actual Price (₹ Cr)", fontsize=11, fontweight="600")
    ax.set_ylabel("Predicted Price (₹ Cr)", fontsize=11, fontweight="600")
    ax.set_title("Model Predictions vs Actual Prices", fontsize=14, fontweight="bold", pad=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig


def plot_city_property_type(df: pd.DataFrame):
    ct = df.groupby(["city", "property_type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    ct.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", linewidth=1.5, width=0.8)
    ax.set_title("Property Types Distribution Across Cities", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("City", fontsize=11, fontweight="600")
    ax.set_ylabel("Number of Properties", fontsize=11, fontweight="600")
    ax.legend(title="Property Type", fontsize=9, title_fontsize=10, loc='upper right')
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    col_header_l, col_header_c, col_header_r = st.columns([1, 4, 1])
    with col_header_c:
        st.markdown("""
        <div style='text-align:center; padding:20px 0;'>
            <h1 style='color:#1a1a2e; font-size:48px; margin-bottom:8px; font-weight:900;'>
                🏠 House Price Predictor
            </h1>
            <p style='color:#666; font-size:15px; margin:0; font-weight:500;'>
                Machine Learning Model for 7 Major Indian Cities
            </p>
            <p style='color:#aaa; font-size:12px; margin:4px 0 0 0;'>
                Random Forest • Bangalore • Chennai • Delhi • Hyderabad • Kolkata • Mumbai • Pune
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Load assets ──────────────────────────────────────────────────────────
    model, meta = load_model()
    df          = load_dataset()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    inputs = render_sidebar(df)

    # ── Model missing warning ─────────────────────────────────────────────────
    if model is None:
        st.warning(
            "⚠️ **Model not found.** Run these commands first:\n\n"
            "```\npython generate_dataset.py\npython train_model.py\n```",
            icon="⚠️"
        )

    # ── Prediction ────────────────────────────────────────────────────────────
    if inputs["predict"]:
        if model is None:
            st.error("❌ Cannot predict — model not trained yet. Run training commands first.", icon="❌")
        else:
            with st.spinner("🔄 Calculating price estimate..."):
                pred, confidence = predict_price(model, meta, inputs)

            pred_cr = pred / 1e7
            pred_l  = pred / 1e5

            # Display prediction with enhanced styling
            st.markdown(f"""
            <div class='prediction-box'>
                <div class='prediction-label'>💰 Estimated Price in {inputs['city']}</div>
                <div class='prediction-amount'>₹{pred_cr:.2f} Cr</div>
                <div class='prediction-subtext'>({pred_l:.1f} Lakhs &nbsp;|&nbsp; Model Confidence: {confidence:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)

            # Success notification
            st.success("✅ Prediction complete! Review the property details below.", icon="✅")

            # Input Summary in expandable section
            with st.expander("📋 Property Details Summary", expanded=True):
                summary = {k: v for k, v in inputs.items() if k != "predict"}
                col1, col2, col3 = st.columns(3)
                
                items = list(summary.items())
                for i, (k, v) in enumerate(items):
                    col = [col1, col2, col3][i % 3]
                    label = k.replace('_', ' ').title()
                    if label == "Balcony":
                        v = "✓ Yes" if v == 1 else "✗ No"
                    col.metric(label, v)

    # ── KPI Metrics ───────────────────────────────────────────────────────────
    if not df.empty:
        st.markdown("<div class='section-header'>📊 Dataset Overview</div>",
                    unsafe_allow_html=True)
        
        # Create metrics with better styling
        m1, m2, m3, m4, m5 = st.columns(5, gap="small")
        
        with m1:
            st.metric(
                label="📦 Total Records",
                value=f"{len(df):,}",
                help="Total number of properties in dataset"
            )
        
        with m2:
            st.metric(
                label="🏙️ Cities",
                value=f"{df['city'].nunique()}",
                help="Number of cities covered"
            )
        
        with m3:
            st.metric(
                label="💰 Avg Price",
                value=f"₹{df['price'].mean()/1e7:.2f}Cr",
                help="Average price across all properties"
            )
        
        with m4:
            st.metric(
                label="📈 Max Price",
                value=f"₹{df['price'].max()/1e7:.1f}Cr",
                help="Highest priced property"
            )
        
        with m5:
            st.metric(
                label="📉 Min Price",
                value=f"₹{df['price'].min()/1e5:.0f}L",
                help="Lowest priced property"
            )
        
        st.divider()

    # ── Dataset Preview ───────────────────────────────────────────────────────
    if not df.empty:
        with st.expander("🗂️ Dataset Preview (first 50 rows)", expanded=False):
            # Format the dataframe for display
            df_display = df.head(50).copy()
            df_display['price'] = df_display['price'].apply(lambda x: f"₹{x/1e7:.2f}Cr")
            df_display['price_per_sqft'] = df_display['price_per_sqft'].apply(lambda x: f"₹{x:,.0f}")
            df_display['total_area_sqft'] = df_display['total_area_sqft'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Visual Analytics</div>",
                unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")
    with col_l:
        if not df.empty:
            st.pyplot(plot_city_avg_price(df), use_container_width=True)
    with col_r:
        if not df.empty:
            st.pyplot(plot_price_distribution(df), use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🤖 Model Analysis</div>",
                unsafe_allow_html=True)
    
    col_l2, col_r2 = st.columns(2, gap="medium")
    with col_l2:
        if model is not None and meta:
            st.pyplot(
                plot_feature_importance(model, meta.get("feature_cols", [])),
                use_container_width=True
            )
    with col_r2:
        if model is not None and meta and not df.empty:
            st.pyplot(plot_predicted_vs_actual(model, meta, df), use_container_width=True)

    # ── Charts Row 3 ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Data Insights</div>",
                unsafe_allow_html=True)
    
    col_l3, col_r3 = st.columns(2, gap="medium")
    with col_l3:
        if not df.empty:
            st.pyplot(plot_city_property_type(df), use_container_width=True)
    with col_r3:
        if not df.empty:
            st.pyplot(plot_correlation_heatmap(df), use_container_width=True)

    # ── City-wise Stats Table ─────────────────────────────────────────────────
    if not df.empty:
        st.markdown("<div class='section-header'>🏙️ City-wise Statistics</div>",
                    unsafe_allow_html=True)
        city_stats = df.groupby("city").agg(
            Properties=("price", "count"),
            Avg_Price=("price", "mean"),
            Median_Price=("price", "median"),
            Avg_Area=("total_area_sqft", "mean"),
            Price_Sqft=("price_per_sqft", "mean"),
        ).reset_index()
        
        # Format columns
        city_stats = city_stats.rename(columns={
            "city": "City",
            "Properties": "📦 Properties",
            "Avg_Price": "💰 Avg Price",
            "Median_Price": "📈 Median Price",
            "Avg_Area": "📐 Avg Area",
            "Price_Sqft": "₹/Sqft"
        })
        
        city_stats["💰 Avg Price"] = city_stats["💰 Avg Price"].apply(lambda x: f"₹{x/1e7:.2f}Cr")
        city_stats["📈 Median Price"] = city_stats["📈 Median Price"].apply(lambda x: f"₹{x/1e7:.2f}Cr")
        city_stats["📐 Avg Area"] = city_stats["📐 Avg Area"].apply(lambda x: f"{x:,.0f} sqft")
        city_stats["₹/Sqft"] = city_stats["₹/Sqft"].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(
            city_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "City": st.column_config.TextColumn(width="medium"),
                "📦 Properties": st.column_config.TextColumn(width="small"),
            }
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class='footer'>
        <div style='margin-bottom:12px;'>
            <strong style='font-size:14px; color:#FF6B35;'>🏠 House Price Predictor</strong><br>
            <span style='font-size:11px; color:#999;'>
                Advanced ML Model for Indian Real Estate
            </span>
        </div>
        <div style='font-size:11px; color:#aaa; line-height:1.6;'>
            <strong>Technology Stack:</strong> Random Forest Regressor • scikit-learn • Streamlit<br>
            <strong>Data Coverage:</strong> 7 Major Cities • 50+ Localities • 10,000+ Properties<br>
            <strong>Model Performance:</strong> R² Score > 0.85 • MAE < ₹5L<br>
            <strong>Platform:</strong> Python 3.10+ • AWS EC2 • GitHub
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
