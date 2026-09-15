# EduGuide AI

An AI-powered academic early-warning platform that predicts student academic risk, explains the reasoning behind each prediction using Explainable AI, and delivers personalized guidance — for students, teachers, and administrators.

## Overview

EduGuide AI follows a Predict → Explain → Guide → Monitor pipeline: a supervised machine learning model classifies students into Low/Medium/High academic risk, SHAP explainability identifies the contributing factors, and a locally-run LLM translates that into natural, role-appropriate guidance for students and teachers alike.

## Features

- **Risk Prediction** — Logistic Regression model, deliberately selected over higher-accuracy alternatives (XGBoost) for its stronger recall on high-risk students
- **Explainable AI** — SHAP-based factor analysis, so predictions are never a black box
- **Personalized Action Plans** — structured immediate/short-term/resource recommendations
- **AI Assistant** — a free, locally-run chatbot (via Ollama) grounded in each student's real data
- **Role-based Dashboards** — separate experiences for Students, Teachers, and Admins
- **Fairness Auditing** — a built-in report confirming the model isn't biased by gender
- **Live Monitoring** — sudden risk-change alerts, live activity feed, and presence tracking

## Tech Stack

- **Backend:** Flask, SQLAlchemy, SQLite
- **Machine Learning:** scikit-learn, SHAP, pandas
- **AI Assistant:** Ollama (local LLM, Llama 3.2)
- **Frontend:** HTML, CSS, JavaScript, Chart.js

## Project Structure

\`\`\`
ML Project/
├── data/           # Raw and processed datasets
├── src/            # Model training and preprocessing scripts
├── models/         # Trained model files
├── database/       # SQLite database
├── backend/        # Flask API
└── *.html          # Frontend pages
\`\`\`

## Setup

1. Clone this repository
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Run `python src/preprocess.py` then `python src/train_model.py`
4. Start the backend: `cd backend && python app.py`
5. Open `index.html` in your browser

## Author

[Keerthi Shri G] — [keerthishri3107@gmail.com]