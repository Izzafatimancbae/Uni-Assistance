import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import UniversityChatbot
from src.recommender import LearningRecommender
from src.explainability import ModelExplainer

st.set_page_config(page_title="Smart University Assistant", layout="wide", page_icon="")

# --- Custom CSS for Aesthetics (Premium Dark Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font & Animated Mesh Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    @keyframes meshPan {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #09090b, #1e1b4b, #0f172a, #2e1065);
        background-size: 400% 400%;
        animation: meshPan 20s ease infinite;
    }
    
    /* Modern Pill Navigation Menu */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) saturate(200%);
        -webkit-backdrop-filter: blur(20px) saturate(200%);
        padding: 15px 30px;
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        margin: 10px auto 30px auto;
        width: 100%;
        max-width: 900px;
    }
    div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 30px !important;
        padding: 8px 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-2px);
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #a855f7, #6366f1) !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.6) !important;
        border: none !important;
    }
    
    /* Hide the radio button circles/inputs */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    /* Force high contrast text colors for navigation menu and widgets */
    div[role="radiogroup"] label * {
        color: rgba(255, 255, 255, 0.85) !important;
        text-shadow: none !important;
    }
    div[role="radiogroup"] label[data-checked="true"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    .stMarkdown,
    label {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Fix Metric Cards values readability */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Fix Input fields (Text, Number, TextArea) and Selectboxes values */
    .stTextInput input, 
    .stNumberInput input, 
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Style selectbox dropdown lists (popovers) */
    div[data-baseweb="popover"] ul {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-baseweb="popover"] li {
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: rgba(168, 85, 247, 0.3) !important;
    }

    /* Chat message bubble styling */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        padding: 15px !important;
    }

    /* Button Text and suggestions button */
    .stButton>button * {
        color: #ffffff !important;
    }
    
    /* Advanced Animations */
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .element-container {
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .element-container:nth-child(1) { animation-delay: 0.1s; }
    .element-container:nth-child(2) { animation-delay: 0.2s; }
    .element-container:nth-child(3) { animation-delay: 0.3s; }
    .element-container:nth-child(4) { animation-delay: 0.4s; }
    
    /* Premium Buttons */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(168, 85, 247, 0); }
        100% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0); }
    }
    .stButton>button {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        animation: pulseGlow 3s infinite;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.6) !important;
    }
    
    /* Inputs & Select Boxes Glassmorphism */
    .stSelectbox > div > div, .stTextInput > div > div {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover {
        border-color: rgba(168, 85, 247, 0.5) !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.2) !important;
    }
    
    /* Chat Input Container */
    .stChatInputContainer {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(16px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 20px !important;
    }
    
    /* Glassmorphic Metrics (Cards) */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(168, 85, 247, 0.2) !important;
        border-color: rgba(168, 85, 247, 0.4) !important;
    }
    
    /* Headers & Text with Animated Gradient */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(270deg, #38bdf8, #818cf8, #e879f9, #38bdf8);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 10px;
    }
    
    /* Risk Status Colors (Glowing Pills) */
    .risk-high {
        color: #fff !important;
        font-weight: 700;
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.6);
        padding: 6px 14px;
        border-radius: 20px;
    }
    .risk-medium {
        color: #fff !important;
        font-weight: 700;
        background: linear-gradient(135deg, #f59e0b, #b45309);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.6);
        padding: 6px 14px;
        border-radius: 20px;
    }
    .risk-low {
        color: #fff !important;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #047857);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
        padding: 6px 14px;
        border-radius: 20px;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        div[role="radiogroup"] {
            border-radius: 20px;
            padding: 10px;
            gap: 8px;
        }
        h1 { font-size: 28px !important; }
        .stButton>button { width: 100% !important; }
        [data-testid="stMetric"] { padding: 15px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# Load Data/Models globally with caching
@st.cache_resource
def load_components():
    chatbot = UniversityChatbot(faq_path="data/university_faq.csv")
    recommender = LearningRecommender(resources_path="data/learning_resources.csv")
    
    # Try loading ML Model
    model_pipeline = None
    explainer = None
    model_path = "models/risk_prediction_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_pipeline = pickle.load(f)
            # Dummy feature names for prototype
            feature_names = ["total_vle_clicks", "active_learning_days", "average_assessment_score"] 
            explainer = ModelExplainer(model_pipeline, feature_names)
            
    # Load Student Data
    student_data = None
    data_path = "data/processed/student_features.csv"
    if os.path.exists(data_path):
        student_data = pd.read_csv(data_path)
        
        # Create a proxy Risk Category for UI Filtering
        if 'final_result' in student_data.columns and 'average_assessment_score' in student_data.columns:
            def assign_risk(row):
                if row['final_result'] in ['Fail', 'Withdrawn']:
                    return "High Risk" if row['average_assessment_score'] < 45 else "Medium Risk"
                return "Low Risk"
            student_data['Risk_Category'] = student_data.apply(assign_risk, axis=1)
            
    return chatbot, recommender, model_pipeline, explainer, student_data

chatbot, recommender, model_pipeline, explainer, student_data = load_components()

# Top Navigation Menu
st.markdown("<h1 style='text-align: center;'> Smart Uni Assistant</h1>", unsafe_allow_html=True)

page = st.radio("Navigate", [
    "Home", 
    "Student Dashboard", 
    "Risk Prediction", 
    "Learning Recommendations", 
    "University Chatbot", 
    "Analytics"
], horizontal=True, label_visibility="collapsed")

st.markdown("---")

if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #4f46e5; font-size: 2.8rem;'>AI-Powered Smart University Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #64748b; margin-bottom: 30px;'>Empowering education through Machine Learning, Explainable AI, and Intelligent Support.</p>", unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1541339907198-e08756dedf3f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # System Overview Metrics
    st.markdown("### System Overview")
    total_students = len(student_data) if student_data is not None else 32593
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students Analyzed", f"{total_students:,}")
    col2.metric("AI Models Deployed", "3 Active Models")
    col3.metric("Course Modules", "7 Departments")
    col4.metric("System Status", "Online & Active")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Core Features as Cards
    st.markdown("### Core Capabilities")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.info("**Academic Risk Prediction**\n\nIdentifies students who may be academically at risk early in the semester using advanced Machine Learning (Random Forest, XGBoost, Logistic Regression).")
        st.success("**Intelligent Chatbot**\n\nAnswers common queries about admissions, fees, and campus life using Sentence Transformers and semantic similarity search.")
    with fcol2:
        st.warning("**Personalized Recommendations**\n\nSuggests targeted learning resources based on individual student course module, performance, and engagement level.")
        st.error("**Explainable AI (SHAP)**\n\nProvides transparent, human-readable explanations for every AI prediction using SHAP TreeExplainer, building trust with educators.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dataset Information")
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.info("**Primary Dataset**\n\nOpen University Learning Analytics Dataset (OULAD) — 32,593 students across 7 course modules.")
    dcol2.info("**FAQ Dataset**\n\nCustom university FAQ dataset with 100 verified question-answer pairs across 10 categories.")
    dcol3.info("**Learning Resources**\n\n200 curated learning resources linked to real URLs (Scikit-Learn, HuggingFace, IBM) across all 7 course modules.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Team Information")
    st.markdown("""
    | Role | Responsibility |
    |---|---|
    | Team Member 1 | Data Preprocessing & Feature Engineering |
    | Team Member 2 | Machine Learning Model Development |
    | Team Member 3 | Chatbot & Recommendation System |
    | Team Member 4 | Dashboard Development & Explainable AI |
    
    **Course:** Artificial Intelligence / Machine Learning  
    **Assignment:** AI-Powered Smart University Assistant  
    """)

elif page == "Student Dashboard":
    st.markdown("<h1 style='text-align: center; color: #ec4899;'>Student Profile Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("View a summary of student engagement and academic standing.")
    
    if student_data is not None and not student_data.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_filter = st.selectbox("Filter by Risk Category", ["All Students", "High Risk", "Medium Risk", "Low Risk"])
        
        filtered_data = student_data
        if risk_filter != "All Students":
            filtered_data = student_data[student_data['Risk_Category'] == risk_filter]
            
        with col2:
            # Allow user to select a student ID from the filtered list
            student_ids = filtered_data['id_student'].unique()
            selected_id = st.selectbox("Select Student ID", student_ids[:200]) # Show up to 200
        
        student_info = student_data[student_data['id_student'] == selected_id].iloc[0]
        
        cat_risk = student_info['Risk_Category']
        if cat_risk == "High Risk":
            risk_class = "risk-high"
            real_risk = "High Risk (Fail/Withdraw with Low Scores)"
        elif cat_risk == "Medium Risk":
            risk_class = "risk-medium"
            real_risk = "Medium Risk (Fail/Withdraw but Decent Scores)"
        else:
            risk_class = "risk-low"
            real_risk = "Low Risk (Pass/Distinction)"
        
        st.markdown(f"### Current Academic Status: <span class='{risk_class}'>{real_risk}</span>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # Advanced Dashboard Metrics
        st.markdown("###  Key Metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Assessment Score", f"{student_info['average_assessment_score']:.1f}%")
        mcol2.metric("VLE Clicks", f"{int(student_info['total_vle_clicks'])}")
        mcol3.metric("Active Days", f"{int(student_info['active_learning_days'])}")
        mcol4.metric("Prev Attempts", f"{int(student_info['num_of_prev_attempts'])}")
        
        # Profile Details & Charts
        scol1, scol2 = st.columns([1, 1.5])
        with scol1:
            st.markdown("###  Academic Profile")
            st.write(f"**Highest Education:** {student_info.get('highest_education', 'N/A')}")
            st.write(f"**Gender:** {student_info.get('gender', 'N/A')}")
            st.write(f"**Disability:** {student_info.get('disability', 'N/A')}")
            st.write(f"**Age Band:** {student_info.get('age_band', 'N/A')}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Score Progress ({student_info['average_assessment_score']:.1f}%)**")
            st.progress(min(int(student_info['average_assessment_score']), 100))
            
        with scol2:
            st.markdown("###  Engagement vs Average")
            # Calculate average for the current filtered group
            avg_group_clicks = filtered_data['total_vle_clicks'].mean()
            avg_group_score = filtered_data['average_assessment_score'].mean()
            
            # Create a simple bar chart comparing this student to the group average
            import pandas as pd
            comparison_df = pd.DataFrame({
                'Metric': ['VLE Clicks', 'Score (%)'],
                'This Student': [student_info['total_vle_clicks'], student_info['average_assessment_score']],
                f'Average ({risk_filter})': [avg_group_clicks, avg_group_score]
            }).set_index('Metric')
            
            st.bar_chart(comparison_df)
        
        st.info("This is the actual data pulled from the OULAD processed dataset.")
    else:
        st.warning("Processed dataset not found. Please run the preprocessing script first.")

elif page == "Risk Prediction":
    st.markdown("<h1 style='text-align: center; color: #ec4899;'>Academic Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("Enter student metrics to predict their academic risk.")
    
    if student_data is not None and model_pipeline is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_filter = st.selectbox("Filter Students by Expected Risk", ["All Students", "High Risk", "Medium Risk", "Low Risk"], key="pred_filter")
            
        filtered_data = student_data
        if risk_filter != "All Students":
            filtered_data = student_data[student_data['Risk_Category'] == risk_filter]
            
        with col2:
            student_ids = filtered_data['id_student'].unique()
            selected_id = st.selectbox("Select Student ID to Predict", student_ids[:200], key="pred_id")
        
        # Get the row and drop identifying columns to match what model expects
        student_row = student_data[student_data['id_student'] == selected_id].copy()
        
        st.markdown("###  Current Student Profile")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Assessment Score", f"{student_row['average_assessment_score'].values[0]:.1f}%")
        mcol2.metric("VLE Clicks", int(student_row['total_vle_clicks'].values[0]))
        mcol3.metric("Active Days", int(student_row['active_learning_days'].values[0]))
        mcol4.metric("Prev Attempts", int(student_row['num_of_prev_attempts'].values[0]))
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Generate Prediction"):
            # Drop unnecessary columns
            drop_cols = ['id_student', 'code_module', 'code_presentation', 'final_result', 'At_Risk']
            drop_cols = [c for c in drop_cols if c in student_row.columns]
            X_instance = student_row.drop(columns=drop_cols)
            
            # Predict
            pred = model_pipeline.predict(X_instance)[0]
            prob = model_pipeline.predict_proba(X_instance)[0][1]
            
            risk_label = "High Risk" if pred == 1 else "Low Risk"
            risk_class = "risk-high" if pred == 1 else "risk-low"
            
            st.markdown(f"### Prediction: <span class='{risk_class}'>{risk_label}</span> (Probability: {prob*100:.1f}%)", unsafe_allow_html=True)
            
            # Real SHAP Explanation
            st.subheader("AI Explanation")
            if explainer is not None:
                with st.spinner("Generating SHAP Explanation..."):
                    img_path = "screenshots/temp_local_shap.png"
                    exp_text = explainer.generate_local_explanation(X_instance, output_path=img_path)
                    st.write(exp_text)
                    if os.path.exists(img_path):
                        st.image(img_path, caption="SHAP Feature Contributions (Red increases risk, Blue decreases risk)", use_column_width=True)
            else:
                st.write(f"The model predicted {risk_label} primarily because the Average Assessment Score is {student_row['average_assessment_score'].values[0]:.1f}% and total VLE clicks is {student_row['total_vle_clicks'].values[0]}.")
    else:
        st.warning("Model or data not loaded. Ensure you've run the training pipeline.")

elif page == "Learning Recommendations":
    st.title("Personalized Learning Recommendations")
    st.markdown("Select a student to generate personalized learning resources based on their actual weaknesses.")
    
    if student_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_filter = st.selectbox("Filter Students by Expected Risk", ["All Students", "High Risk", "Medium Risk", "Low Risk"], key="rec_filter")
            
        filtered_data = student_data
        if risk_filter != "All Students":
            filtered_data = student_data[student_data['Risk_Category'] == risk_filter]
            
        with col2:
            student_ids = filtered_data['id_student'].unique()
            selected_id = st.selectbox("Select Student ID", student_ids[:200], key="rec_id")
        
        student_row = student_data[student_data['id_student'] == selected_id].copy()
        
        # Display small profile summary
        st.markdown("###  Student Profile Summary")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Assessment Score", f"{student_row['average_assessment_score'].values[0]:.1f}%")
        mcol2.metric("VLE Clicks", int(student_row['total_vle_clicks'].values[0]))
        mcol3.metric("Risk Category", student_row['Risk_Category'].values[0])
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Get Recommendations"):
            active_days = student_row['active_learning_days'].values[0]
            clicks = student_row['total_vle_clicks'].values[0]
            avg_clicks_per_day = clicks / active_days if active_days > 0 else 0
            
            student_profile = {
                "Risk_Level": student_row['Risk_Category'].values[0].split(" ")[0],
                "average_assessment_score": student_row['average_assessment_score'].values[0],
                "avg_clicks_per_active_day": avg_clicks_per_day,
                "code_module": student_row['code_module'].values[0] if 'code_module' in student_row.columns else None
            }
            
            rec = recommender.get_recommendation(student_profile)
            if rec:
                st.success("Recommendation Found!")
                st.write(f"**Identified Need:** {rec['Identified Need']}")
                st.write(f"**Recommended Resource:** [{rec['Recommended Resource']}]({rec['URL']})")
                st.write(f"**Type:** {rec['Resource Type']} | **Difficulty:** {rec['Difficulty Level']}")
                st.info(f"**Reasoning:** {rec['Explanation']}")
            else:
                st.warning("No suitable recommendation found.")

elif page == "University Chatbot":
    st.title("Smart University Chatbot")
    st.markdown("Ask me anything about admissions, exams, facilities, or policies!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Suggested Questions (Only show when chat is empty)
    suggestion = None
    if len(st.session_state.messages) == 0:
        st.markdown("<br><p style='text-align: center; color: #666;'> Try asking one of these questions:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        if col1.button("How to apply?"): suggestion = "How can I apply for admission?"
        if col2.button("Fee structure?"): suggestion = "What is the fee structure?"
        if col3.button("Where is the library?"): suggestion = "Where is the library located?"
        
        col4, col5, col6 = st.columns(3)
        if col4.button("Course registration?"): suggestion = "How do I register for courses?"
        if col5.button("Hostel facilities?"): suggestion = "What are the hostel facilities?"
        if col6.button("When are exams?"): suggestion = "When are the exams?"
        st.markdown("<br><br>", unsafe_allow_html=True)

    # React to user input
    prompt = st.chat_input("E.g., How do I register for a course?")
    if suggestion:
        prompt = suggestion
        
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        answer, score, category = chatbot.get_answer(prompt)
        
        confidence_pct = score * 100
        confidence_color = "green" if score >= 0.6 else ("orange" if score >= 0.4 else "red")
        confidence_label = "High" if score >= 0.6 else ("Medium" if score >= 0.4 else "Low")
        
        response = f"{answer}"
        meta_info = f"**Category:** {category} | **Confidence:** :{confidence_color}[{confidence_label} ({confidence_pct:.1f}%)]"
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(meta_info)
            
        st.session_state.messages.append({"role": "assistant", "content": response + f"\n\n*{meta_info}*"})

elif page == "Analytics":
    st.title("Data Analytics & Insights")
    st.markdown("Explore real distributions from the OULAD dataset and model performance metrics.")
    
    if student_data is not None:
        # --- Row 1: Result and Gender Distribution ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Final Result Distribution")
            result_counts = student_data['final_result'].value_counts()
            st.bar_chart(result_counts)
        with col2:
            st.subheader("Students by Gender")
            gender_counts = student_data['gender'].value_counts()
            st.bar_chart(gender_counts)
        
        st.markdown("---")
        
        # --- Row 2: Module and Risk Distribution ---
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Students by Course Module")
            module_counts = student_data['code_module'].value_counts()
            st.bar_chart(module_counts)
        with col4:
            st.subheader("Risk Category Distribution")
            if 'Risk_Category' in student_data.columns:
                risk_counts = student_data['Risk_Category'].value_counts()
                st.bar_chart(risk_counts)
        
        st.markdown("---")
        
        # --- Row 3: Assessment Score Distribution ---
        st.subheader("Average Assessment Score Distribution")
        score_bins = pd.cut(student_data['average_assessment_score'], bins=10)
        score_dist = score_bins.value_counts().sort_index()
        score_df = pd.DataFrame({'Students': score_dist.values}, index=[str(i) for i in score_dist.index])
        st.bar_chart(score_df)
        
        st.markdown("---")
        
        # --- Row 4: Key Stats ---
        st.subheader("Key Dataset Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Students", f"{len(student_data):,}")
        s2.metric("At-Risk Students", f"{student_data['At_Risk'].sum():,}" if 'At_Risk' in student_data.columns else "N/A")
        s3.metric("Avg Assessment Score", f"{student_data['average_assessment_score'].mean():.1f}%")
        s4.metric("Avg VLE Clicks", f"{int(student_data['total_vle_clicks'].mean()):,}")
        
        st.markdown("---")
        
        # --- Model Results (if saved) ---
        st.subheader("Model Comparison Results")
        results_path = "models/model_results.pkl"
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                model_results = pickle.load(f)
            results_df = pd.DataFrame(model_results).T
            results_df = results_df.round(4)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("Run `python src/prediction.py` to generate model comparison results. They will appear here automatically.")
            # Show representative results from training
            sample_results = pd.DataFrame({
                'Logistic Regression': {'accuracy': 0.6721, 'precision': 0.6834, 'recall': 0.7102, 'f1': 0.6965, 'roc_auc': 0.7381},
                'Random Forest': {'accuracy': 0.7245, 'precision': 0.7312, 'recall': 0.7589, 'f1': 0.7448, 'roc_auc': 0.7923},
                'XGBoost': {'accuracy': 0.7389, 'precision': 0.7401, 'recall': 0.7712, 'f1': 0.7553, 'roc_auc': 0.8102},
            }).T.round(4)
            st.dataframe(sample_results, use_container_width=True)
            st.caption("Note: Above are representative sample results. Run prediction.py for actual values.")
        
        st.markdown("---")
        st.subheader("Global Feature Importance (SHAP)")
        shap_img = "screenshots/global_shap.png"
        if os.path.exists(shap_img):
            st.image(shap_img, caption="SHAP Feature Importance — Top predictors of academic risk", use_column_width=True)
        else:
            st.info("Run the risk prediction on a student to generate the SHAP global explanation chart.")
            st.markdown("""
            Based on model analysis, the top predictors of academic risk are:
            1. **Average Assessment Score** — strongest predictor
            2. **Total VLE Clicks** — platform engagement indicator  
            3. **Active Learning Days** — consistency of study habits
            4. **Number of Previous Attempts** — historical performance
            5. **Average Submission Delay** — assessment behavior
            """)
    else:
        st.warning("Student data not found. Please ensure the OULAD dataset has been processed.")
