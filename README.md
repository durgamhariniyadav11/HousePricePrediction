# 🏠 Cloud-Based House Price Prediction System — Indian Real Estate (7 Cities)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![AWS](https://img.shields.io/badge/AWS-EC2-yellow?logo=amazon-aws)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A complete end-to-end Machine Learning web application that predicts house prices across **7 major Indian cities** (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune) using a **Random Forest Regressor**. The app features a modern **Streamlit** dashboard with interactive charts, city-wise analytics, and real-time predictions — deployable locally or on **AWS EC2**.

**Dataset:** [Real Estate Data from 7 Indian Cities](https://www.kaggle.com/datasets/rakkesharv/real-estate-data-from-7-indian-cities) — scraped from Indian real estate portals, hosted on GitHub.

---

## 🧰 Technologies Used

| Category        | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.10+                        |
| ML Framework    | scikit-learn (RandomForestRegressor)|
| Data Processing | pandas, NumPy                       |
| Visualization   | Matplotlib, Seaborn                 |
| Web Framework   | Streamlit                           |
| Model Persistence | joblib                            |
| Cloud Platform  | AWS EC2 (Ubuntu 22.04)              |
| Version Control | Git & GitHub                        |

---

## 📁 Project Structure

```
HousePricePrediction/
│
├── app.py                 ← Streamlit web application
├── generate_dataset.py    ← Downloads & cleans dataset from GitHub
├── train_model.py         ← Data preprocessing + model training script
├── test_model.py          ← Model evaluation & accuracy report
├── dataset.csv            ← Cleaned Indian housing dataset (7 cities)
├── model_meta.pkl         ← Saved encoders + feature names (tracked in git)
├── requirements.txt       ← Python dependencies
├── Dockerfile             ← Docker image (auto-generates model.pkl at build)
├── .dockerignore          ← Docker build exclusions
├── README.md              ← Project documentation
├── .gitignore             ← Git ignore rules
└── .streamlit/
    └── config.toml        ← Streamlit server configuration
```

> ⚠️ **`model.pkl` is NOT included in the repository** (105 MB — exceeds GitHub's 25 MB limit).
> It is generated automatically by running `python train_model.py` or during `docker build`.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/HousePricePrediction.git
cd HPP
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download & Generate Dataset

```bash
python generate_dataset.py
```

This downloads the **7-city Indian Real Estate dataset** from GitHub and saves `dataset.csv`.

### Step 5 — Train the Model

```bash
python train_model.py
```

> ⚠️ `model.pkl` is **not included in the repo** (too large for GitHub). You must run this step to generate it before launching the app.

Expected output:
```
[INFO] Dataset loaded: 14,262 rows × 9 columns
[INFO] Training RandomForestRegressor …
[INFO] Training complete.
  MAE  : ₹716,770
  RMSE : ₹2,680,123
  R²   : 0.9274
[INFO] Model saved → 'model.pkl'
```

### Step 6 — Launch the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🚀 How to Run Locally (Quick Start)

```bash
# One-liner after setup
python generate_dataset.py && python train_model.py && streamlit run app.py
```

---

## ☁️ AWS EC2 Deployment Steps

### 1. Launch an EC2 Instance
- Go to **AWS Console → EC2 → Launch Instance**
- Choose **Ubuntu Server 22.04 LTS (Free Tier eligible)**
- Instance type: `t2.micro` (free tier) or `t2.small`
- Configure Security Group — add **Inbound Rule**:
  - Type: Custom TCP | Port: **8501** | Source: 0.0.0.0/0

### 2. Connect to EC2 via SSH

```bash
ssh -i "your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

### 3. Install Python & Git on EC2

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

### 4. Clone Your Repository on EC2

```bash
git clone https://github.com/YOUR_USERNAME/HousePricePrediction.git
cd HousePricePrediction
```

> Make sure you have pushed your project to GitHub first (see the GitHub section above).

### 5. Set Up Virtual Environment & Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 6. Download Dataset & Train the Model

```bash
python3 generate_dataset.py
python3 train_model.py
```

> ⚠️ `model.pkl` is not in the repo. This step generates it (takes ~1–2 min).

### 7. Run Streamlit on EC2

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 8. Access the App

Open in browser:
```
http://YOUR_EC2_PUBLIC_IP:8501
```

### 9. Keep App Running (Background Process)

```bash
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

To stop it:
```bash
pkill -f streamlit
```

---

## 📤 GitHub — Push Your Project (First Time)

> ⚠️ **Do NOT clone your own project.** You push it first, then clone on other machines.

### Step 1 — Create a new empty repo on GitHub
1. Go to [github.com/new](https://github.com/new)
2. Name it `HousePricePrediction`
3. Set visibility to **Public**
4. **Leave all checkboxes unchecked** (no README, no .gitignore)
5. Click **Create repository**

### Step 2 — Push from your local machine

```bash
git init
git add .
git commit -m "Initial commit: Indian House Price Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/HousePricePrediction.git
git push -u origin main
```

### Step 3 — Subsequent pushes (after changes)

```bash
git add .
git commit -m "Update: <describe your change>"
git push
```

### Step 4 — Clone on a NEW machine (e.g. AWS EC2)

Only after pushing, you can clone on another machine:

```bash
git clone https://github.com/YOUR_USERNAME/HousePricePrediction.git
cd HousePricePrediction
```

---

## � Docker

`model.pkl` is generated **automatically during `docker build`** — no manual training needed.

```bash
# Build (trains model inside the image — takes ~2 min)
docker build -t hpp-app .

# Run
docker run -p 8501:8501 hpp-app

# Open in browser
# http://localhost:8501
```

| Metric | Description                          |
|--------|--------------------------------------|
| MAE    | Mean Absolute Error — avg dollar error|
| RMSE   | Root Mean Squared Error              |
| R²     | Coefficient of Determination (0–1)   |

---

## 🖼️ Screenshots

> Add screenshots of your running app here after deployment.

| Dashboard | Prediction Result |
|-----------|-------------------|
| *(screenshot)* | *(screenshot)* |

---

## 🔮 Future Enhancements

- [ ] Integrate full California Housing dataset (20,000+ records) from Kaggle
- [ ] Add XGBoost and LightGBM model comparison
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Add user authentication with AWS Cognito
- [ ] Store predictions in AWS RDS / DynamoDB
- [ ] Deploy using AWS Elastic Beanstalk or Docker + ECS
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Add map visualization using Folium / Plotly

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
  🚀 <strong>Developed using Machine Learning and AWS Cloud</strong>
</div>
