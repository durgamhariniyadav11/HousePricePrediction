# =============================================================================
# Dockerfile
# Cloud-Based House Price Prediction — Indian Real Estate (7 Cities)
# Streamlit app on Python 3.12 slim image
# Build : docker build -t hpp-app .
# Run   : docker run -p 8501:8501 hpp-app
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="HPP Project"
LABEL description="Indian House Price Prediction — Streamlit + Random Forest"

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1  → required by scikit-learn / OpenMP (RandomForest parallel jobs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer-cache friendly) ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py             .
COPY generate_dataset.py .
COPY train_model.py     .
COPY test_model.py      .
COPY .streamlit/        .streamlit/

# ── Copy dataset and model metadata ──────────────────────────────────────────
# dataset.csv and model_meta.pkl are tracked in git and copied directly.
# model.pkl (~105MB) is NOT in the repo — it is generated at build time below.
COPY dataset.csv        ./dataset.csv
COPY model_meta.pkl     ./model_meta.pkl

# ── Generate model.pkl at build time ─────────────────────────────────────────
# Runs train_model.py inside the image so model.pkl is baked in.
# dataset.csv must already be present (copied above).
RUN python train_model.py

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
