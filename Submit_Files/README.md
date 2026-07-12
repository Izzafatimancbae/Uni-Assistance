# AI-Powered Smart University Assistant

A complete prototype AI system that integrates **Machine Learning**, **Natural Language Processing**, **Explainable AI**, and **Personalized Recommendations** into a single Streamlit web application to support university students and academic advisors.

---

## System Overview

| Component | Technology | Purpose |
|---|---|---|
| Risk Prediction | XGBoost, Random Forest, Logistic Regression | Predict at-risk students |
| Chatbot | Sentence Transformers (all-MiniLM-L6-v2) | Answer university FAQs |
| Recommender | Rule-Based System | Course-specific resource recommendations |
| Explainable AI | SHAP TreeExplainer | Transparent model explanations |
| Dashboard | Streamlit | Interactive web application |

---

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

- **32,593** student records across **7 course modules** (AAA–GGG)
- Files: `studentInfo.csv`, `studentAssessment.csv`, `assessments.csv`, `studentVle.csv`, `studentRegistration.csv`
- Download from: [https://analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset)

Place downloaded files in: `data/raw/`

---

## Project Structure

```
smart-university-assistant/
├── data/
│   ├── raw/                    ← Place OULAD CSV files here
│   ├── processed/              ← Auto-generated processed features
│   ├── university_faq.csv      ← 100 university FAQ records
│   └── learning_resources.csv  ← 200 learning resources (real URLs)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_explainable_ai.ipynb
├── models/
│   ├── risk_prediction_model.pkl
│   └── early_risk_prediction_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── chatbot.py
│   ├── recommender.py
│   └── explainability.py
├── app/
│   └── app.py
├── screenshots/
├── report/
│   └── final_report.md
├── README.md
└── requirements.txt
```

---

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Place OULAD Dataset

Download all CSV files from the OULAD website and place them in `data/raw/`:
- `studentInfo.csv`
- `studentAssessment.csv`
- `assessments.csv`
- `studentVle.csv`
- `studentRegistration.csv`
- `courses.csv`

### 3. Run Data Preprocessing

```bash
python src/preprocessing.py
```

This generates `data/processed/student_features.csv` and `data/processed/early_warning_features.csv`.

### 4. Train Machine Learning Models

```bash
python src/prediction.py
```

This trains Logistic Regression, Random Forest (tuned), and XGBoost (tuned) with 5-fold cross-validation and hyperparameter tuning. The best model is saved to `models/risk_prediction_model.pkl`.

### 5. Generate Datasets (FAQ + Learning Resources)

```bash
python generate_datasets.py
```

### 6. Launch the Dashboard

```bash
streamlit run app/app.py
```

---

## Application Pages

| Page | Features |
|---|---|
| **Home** | Project overview, dataset statistics, team information |
| **Student Dashboard** | Profile, assessment score, engagement metrics, risk category |
| **Risk Prediction** | Select student, generate prediction, view SHAP explanation |
| **Learning Recommendations** | Course-specific resources matched to student needs |
| **University Chatbot** | Semantic search, confidence score, category, fallback |
| **Analytics** | Real data charts, model comparison table, feature importance |

---

## Key Results

- **Best Model:** XGBoost (Tuned) — F1-Score: ~0.755, ROC-AUC: ~0.810
- **Chatbot Accuracy:** 90% (27/30 test queries answered correctly)
- **Early Warning (30-day):** ~68% F1 — enables intervention at week 4–5

---

## Ethical Considerations

- All student data is anonymized (no personally identifiable information)
- SHAP explanations ensure model transparency for advisors
- Predictions are decision-support tools only — final decisions require human judgment
- Model fairness across demographic groups should be evaluated before production use

---

## Requirements

```
pandas==2.2.0
numpy==1.26.4
matplotlib==3.8.2
scikit-learn==1.4.0
xgboost==2.0.3
shap==0.44.1
sentence-transformers==2.3.1
streamlit==1.31.0
jupyter==1.0.0
```
