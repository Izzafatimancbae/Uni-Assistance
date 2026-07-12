# AI-Powered Smart University Assistant
## Final Project Report

**Course:** Artificial Intelligence / Machine Learning  
**Assignment:** Group Practical Assignment — AI-Powered Smart University Assistant  
**Total Marks:** 100  

---

## Title Page

**Project Title:** AI-Powered Smart University Assistant  
**Team Members:**
| Name | Student ID | Role |
|---|---|---|
| Team Member 1 | [ID] | Data Preprocessing & Feature Engineering |
| Team Member 2 | [ID] | Machine Learning Model Development |
| Team Member 3 | [ID] | Chatbot & Recommendation System |
| Team Member 4 | [ID] | Dashboard Development & Explainable AI |

**Submission Date:** [Date]  
**Generative AI Declaration:** GitHub Copilot and Antigravity AI coding assistant were used to assist in code development. All AI outputs were reviewed, verified, and adapted by team members.

---

## Abstract

This report presents the design and development of an AI-Powered Smart University Assistant, a prototype system that integrates machine learning, natural language processing, a recommendation engine, and explainable AI into a single interactive web application. Using the Open University Learning Analytics Dataset (OULAD) containing 32,593 student records, the system predicts academic risk, recommends personalized learning resources, answers university-related queries through an intelligent chatbot, and provides transparent explanations for every prediction. The system achieves an F1-score of approximately 0.75 on the full-semester risk model and demonstrates that early identification of at-risk students using only 30 days of data is feasible, albeit with reduced accuracy.

---

## 1. Introduction

The rapid digitization of educational environments has produced vast quantities of student behavioral data that institutions have yet to leverage effectively. Academic advisors continue to identify struggling students through manual observation, often too late in the semester for meaningful intervention. This project addresses this gap by developing an intelligent university assistant that automatically analyzes student learning data and provides actionable insights to both students and educators.

The system integrates five areas of artificial intelligence:
1. Machine Learning for academic risk classification
2. Natural Language Processing for FAQ retrieval
3. A rule-based recommendation system for learning resources
4. Explainable AI through SHAP analysis
5. An interactive Streamlit dashboard for intelligent decision support

---

## 2. Problem Statement

A university aims to improve student support services using artificial intelligence. The key challenges are:
- Academic advisors identify weak students manually, often mid or late semester
- Students struggle to find relevant course information
- Learning resources are not personalized to individual performance
- Prediction systems are "black boxes" that advisors do not trust

**Stakeholders:**
- **Students** — primary beneficiaries of risk warnings and resource recommendations
- **Academic Advisors** — users of the risk prediction and SHAP explanation tools
- **University Administration** — interested in systemic patterns and at-risk rates
- **IT Department** — responsible for maintaining the deployed application

---

## 3. Assignment Objectives

After completing this system, the team demonstrated ability to:
- Analyze the OULAD real-world educational dataset
- Clean, transform, and integrate multiple data files
- Apply exploratory data analysis across 8+ visualizations
- Develop and compare three machine learning classification models
- Build an NLP-based question-answering chatbot using Sentence Transformers
- Design a rule-based recommendation system linked to course modules
- Apply SHAP explainable AI techniques globally and locally
- Develop a fully integrated Streamlit web application
- Evaluate ethical and privacy implications of educational AI

---

## 4. Related Work

Educational data mining and learning analytics have emerged as significant research areas. Romero and Ventura (2010) surveyed early applications of data mining to e-learning, establishing foundational patterns for predicting student performance. Aljohani et al. (2019) demonstrated that VLE interaction data from OULAD is particularly predictive of student outcomes, with ensemble methods outperforming traditional classifiers. More recently, Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), which has become the standard approach for explaining tree-based model predictions in high-stakes domains such as education.

---

## 5. Dataset Description

### 5.1 Primary Dataset: OULAD

The Open University Learning Analytics Dataset (OULAD) provides anonymized student records from the UK's Open University.

| File | Records | Description |
|---|---|---|
| studentInfo.csv | 32,593 | Demographics, registration, final result |
| studentAssessment.csv | 173,912 | Assessment scores per student |
| assessments.csv | 206 | Assessment metadata |
| studentVle.csv | ~10M | VLE click interactions |
| studentRegistration.csv | 32,593 | Registration and withdrawal dates |

**Target Variable:** Binary classification — `At_Risk = 1` if final_result is 'Fail' or 'Withdrawn', `At_Risk = 0` if 'Pass' or 'Distinction'.

**Class Distribution:**
- At Risk: 17,208 (52.8%)
- Not At Risk: 15,385 (47.2%)

### 5.2 FAQ Dataset

A custom `university_faq.csv` file was created containing 100 verified question-answer pairs across 10 categories: Admissions, Fee and Finance, Course Registration, Attendance, Examinations, Scholarships, Internships, Final-Year Projects, Library Services, and Campus Facilities.

### 5.3 Learning Resources Dataset

A `learning_resources.csv` file was created with 200 resources across all 7 course modules (AAA–GGG) and 3 difficulty levels, with real URLs linking to Scikit-Learn documentation, HuggingFace NLP courses, IBM, and other educational platforms.

---

## 6. Data Preprocessing

### 6.1 Data Cleaning
- `imd_band` NaN values (approximately 1%) were filled with the category 'Missing'
- `studentAssessment.score` NaN values were filled with 0 (students who did not score)
- No duplicate student records were identified
- `date_unregistration` NaN values were coded as 999 (indicating no withdrawal)

### 6.2 Feature Engineering

Seven features were engineered from multiple OULAD files:

| Feature | Source | Description |
|---|---|---|
| total_vle_clicks | studentVle | Total number of learning platform interactions |
| active_learning_days | studentVle | Number of unique days with VLE activity |
| avg_clicks_per_active_day | Derived | Engagement intensity metric |
| assessments_attempted | studentAssessment | Number of assessments submitted |
| average_assessment_score | studentAssessment | Mean assessment score |
| avg_submission_delay | studentAssessment + assessments | Average days before/after deadline |
| avg_registration_delay | studentRegistration | Registration timing relative to course start |

### 6.3 Categorical Encoding
One-Hot Encoding was applied to categorical features: `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`. All numerical features were standardized using StandardScaler within sklearn Pipeline objects.

---

## 7. Exploratory Data Analysis

Eight visualizations were produced, revealing the following key patterns:

1. **Result Distribution:** Pass (37.9%), Withdrawn (31.2%), Fail (21.6%), Distinction (9.3%)
2. **Gender:** Male students have a marginally higher at-risk rate than female students
3. **Education Level:** Students with lower prior qualifications show higher at-risk rates
4. **Assessment Scores:** At-risk students average ~38% vs ~82% for not-at-risk students
5. **VLE Activity:** At-risk students generate significantly fewer VLE clicks on average
6. **Engagement vs Score:** Strong positive correlation between engagement and performance
7. **Correlation Matrix:** `average_assessment_score` has the strongest negative correlation with `At_Risk` (-0.62)
8. **Class Distribution:** Near-balanced classes (52.8% vs 47.2%) — minimal imbalance correction needed

---

## 8. Machine Learning Models

### 8.1 Model Architecture

Three algorithms were trained and compared using an sklearn Pipeline:

| Model | Type | Imbalance Strategy |
|---|---|---|
| Logistic Regression | Linear | class_weight='balanced' |
| Random Forest | Ensemble (Bagging) | class_weight='balanced' |
| XGBoost | Ensemble (Boosting) | scale_pos_weight |

### 8.2 Training Procedure
- **Train/Test Split:** 80/20 with `stratify=y` (random_state=42)
- **Cross-Validation:** 5-Fold StratifiedKFold on training set
- **Hyperparameter Tuning:** GridSearchCV (3-fold) for Random Forest and XGBoost

### 8.3 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.672 | 0.683 | 0.710 | 0.697 | 0.738 |
| Random Forest (Tuned) | 0.724 | 0.731 | 0.759 | 0.745 | 0.792 |
| XGBoost (Tuned) | 0.739 | 0.740 | 0.771 | 0.755 | 0.810 |

**Final Model:** XGBoost (Tuned) selected based on highest F1-score and ROC-AUC.

### 8.4 Why Recall Matters

In academic risk prediction, a False Negative (missing an at-risk student) has greater consequences than a False Positive (unnecessary advisor check-in). Therefore, the model selection prioritizes **Recall** and **F1-score** over raw accuracy.

---

## 9. Early Warning Prediction

An early warning model was trained using only VLE and assessment data from the **first 30 days** of the course. This simulates predicting risk before most assessments are completed.

| Metric | Full Semester | Early Warning (30-day) |
|---|---|---|
| F1-Score | ~0.755 | ~0.68 |
| Recall | ~0.771 | ~0.71 |
| ROC-AUC | ~0.810 | ~0.73 |

**Finding:** The early warning model can identify approximately 71% of at-risk students using only the first 30 days of data — sufficient for meaningful early intervention while accepting a slight reduction in accuracy.

---

## 10. Chatbot Design

The University FAQ Chatbot uses a semantic retrieval approach:

1. **Text Preprocessing:** Lowercase conversion, punctuation removal
2. **Embedding Model:** `all-MiniLM-L6-v2` from Sentence Transformers
3. **Retrieval Method:** Cosine similarity between user query embedding and FAQ embeddings
4. **Confidence Threshold:** 0.5 — queries below this threshold trigger the fallback message
5. **Fallback Message:** *"I could not find a sufficiently reliable answer. Please contact the relevant university office or rephrase your question."*
6. **Unanswered Logging:** All fallback queries are logged to `data/unanswered_questions.txt`

### Chatbot Evaluation (30 Test Cases)

| Category | Questions | Correct | Accuracy |
|---|---|---|---|
| FAQ-similar | 15 | 14 | 93.3% |
| Paraphrased | 10 | 8 | 80.0% |
| Out-of-scope | 5 | 5 (fallback) | 100% |
| **Overall** | **30** | **27** | **90.0%** |

Average similarity score on correct answers: **0.74**

---

## 11. Recommendation System

The recommendation system uses **Rule-Based Recommendation (Option A)**:

| Rule | Condition | Difficulty | Resource Type |
|---|---|---|---|
| 1 | High risk + score < 50% | Beginner | Tutorial/Video |
| 2 | High/Medium risk + low engagement | Beginner | Short Video |
| 3 | Medium risk + score 50–70% | Intermediate | Practice Exercise |
| 4 | Low risk + score ≥ 70% | Advanced | Research Paper |
| 5 | Default | Intermediate | Article |

Resources are filtered by **both the student's course module** (AAA–GGG) and recommended difficulty, ensuring relevance. Real URLs from Scikit-Learn, HuggingFace, IBM, and MachineLearningMastery are provided.

---

## 12. Explainable AI

### Global Explanation (SHAP)
The SHAP TreeExplainer generates a summary plot showing the global importance of each feature across all predictions. The top 5 predictors are:
1. Average Assessment Score — strongest negative predictor of risk
2. Total VLE Clicks — higher engagement = lower risk
3. Active Learning Days — consistency matters
4. Number of Previous Attempts — repeat attempts correlate with risk
5. Average Submission Delay — late submissions are a risk signal

### Local Explanation
For each individual student, a SHAP waterfall plot is generated showing which features pushed the prediction toward or away from At-Risk. A human-readable text explanation is also displayed in the dashboard.

---

## 13. Application Development

The application was built with Streamlit and includes 6 pages:

| Page | Key Features |
|---|---|
| Home | Project overview, dataset stats, team info, core capabilities |
| Student Dashboard | Student profile, assessment score, VLE clicks, risk category |
| Risk Prediction | Select student, generate ML prediction, probability, SHAP explanation |
| Learning Recommendations | Course-specific resources, difficulty-matched, real URLs |
| University Chatbot | Semantic search, confidence score, category display, fallback |
| Analytics | Real data charts (result/gender/module distributions), model comparison |

---

## 14. Results and Discussion

The system successfully demonstrates practical AI integration for educational support. XGBoost achieved the best balance of precision and recall. The chatbot performs well on expected questions but predictably struggles with highly paraphrased or ambiguous queries. The recommendation system appropriately tailors suggestions to both student performance and course module.

A key limitation is that the recommendation rules are manually defined rather than learned from data — a hybrid or collaborative filtering approach would improve personalization in a production system.

---

## 15. Ethical Considerations

### Student Privacy
All student data uses anonymized IDs (no real names or personally identifiable information). In a production deployment, GDPR and local data protection laws must be followed. Explicit informed consent must be obtained before collecting or processing student data.

### Algorithmic Bias
The model was trained on students from the UK Open University. Predictions may not generalize to students from different cultural, economic, or educational backgrounds. The `imd_band` (deprivation index) feature is used for prediction, which raises concerns about reinforcing socioeconomic disadvantage.

### Fairness
Gender is included as a feature. While it may improve predictive accuracy, it risks encoding historical biases. Future work should evaluate model fairness across demographic groups.

### Misclassification Risk
False negatives (missing at-risk students) and false positives (unnecessary interventions) both have real costs. The system must be positioned as a **decision-support tool** — final academic decisions must always involve human judgment.

### Human Oversight
Academic advisors should review all AI recommendations before taking action. The SHAP explanations are designed to facilitate this oversight by making model reasoning transparent and understandable to non-technical users.

---

## 16. Limitations

1. The chatbot's knowledge is limited to the university FAQ dataset — it cannot answer general academic questions
2. The recommendation system uses fixed rules rather than learned preferences
3. The early warning model has reduced accuracy due to limited data in the first 30 days
4. The system has not been tested with real students in a live university environment
5. Model drift (performance degradation over time) is not currently monitored

---

## 17. Conclusion

This project successfully developed an integrated AI University Assistant that demonstrates the practical application of five areas of artificial intelligence. The system is capable of predicting academic risk with ~75% F1-score, answering 90% of university FAQ queries correctly, and generating course-specific learning recommendations with transparent SHAP explanations. The Streamlit dashboard provides an accessible interface for both students and academic advisors.

Future work should focus on: (1) replacing rule-based recommendations with a content-based or collaborative filtering approach, (2) deploying the system with real student data under proper privacy governance, and (3) implementing continuous model monitoring to detect performance degradation over time.

---

## 18. References

1. Romero, C., & Ventura, S. (2010). Educational Data Mining: A Review of the State of the Art. *IEEE Transactions on Systems, Man, and Cybernetics*, 40(6), 601–618.
2. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data*, 4, 170171.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.

---

## Appendix A: Project Folder Structure

```
smart-university-assistant/
├── data/
│   ├── raw/           (OULAD files)
│   ├── processed/     (student_features.csv, early_warning_features.csv)
│   ├── university_faq.csv
│   └── learning_resources.csv
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
├── app/app.py
├── screenshots/
├── report/
├── README.md
└── requirements.txt
```

---

## Appendix B: Individual Contribution Statement

| Team Member | Contribution | Percentage |
|---|---|---|
| [Member 1] | Data collection, preprocessing, EDA notebooks | 25% |
| [Member 2] | Model training, evaluation, hyperparameter tuning | 25% |
| [Member 3] | Chatbot development, recommendation system | 25% |
| [Member 4] | Dashboard development, SHAP integration, report | 25% |

*All team members reviewed and verified each other's work.*
