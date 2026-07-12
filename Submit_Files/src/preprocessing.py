import pandas as pd
import numpy as np
import os

def load_data(raw_data_dir="data/raw"):
    """Loads the OULAD dataset files."""
    print("Loading datasets...")
    try:
        student_info = pd.read_csv(os.path.join(raw_data_dir, "studentInfo.csv"))
        student_assessment = pd.read_csv(os.path.join(raw_data_dir, "studentAssessment.csv"))
        assessments = pd.read_csv(os.path.join(raw_data_dir, "assessments.csv"))
        student_vle = pd.read_csv(os.path.join(raw_data_dir, "studentVle.csv"))
        student_registration = pd.read_csv(os.path.join(raw_data_dir, "studentRegistration.csv"))
        return student_info, student_assessment, assessments, student_vle, student_registration
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure OULAD files are in {raw_data_dir}.")
        return None, None, None, None, None

def preprocess_and_engineer_features(student_info, student_assessment, assessments, student_vle, student_registration):
    """
    Cleans data, engineers features, and merges datasets into a single dataframe for modeling.
    """
    print("Preprocessing data and engineering features...")
    
    # 1. Target Definition
    # Binary Target: 1 for Fail or Withdrawn (At Risk), 0 for Pass or Distinction (Not At Risk)
    student_info['At_Risk'] = student_info['final_result'].map({'Fail': 1, 'Withdrawn': 1, 'Pass': 0, 'Distinction': 0})
    
    # Handle missing values in student_info (imd_band)
    student_info['imd_band'].fillna('Missing', inplace=True)
    
    # 2. VLE (Virtual Learning Environment) Feature Engineering
    # Aggregate clicks per student
    vle_agg = student_vle.groupby('id_student').agg(
        total_vle_clicks=('sum_click', 'sum'),
        active_learning_days=('date', 'nunique')
    ).reset_index()
    
    vle_agg['avg_clicks_per_active_day'] = vle_agg['total_vle_clicks'] / vle_agg['active_learning_days']
    vle_agg['avg_clicks_per_active_day'].fillna(0, inplace=True)
    
    # 3. Assessment Feature Engineering
    # Merge student assessment with assessment details to get weight and type
    assessment_merged = pd.merge(student_assessment, assessments, on='id_assessment', how='left')
    
    # Handle missing scores (assume 0 if not scored but submitted, though rare in this dataset)
    assessment_merged['score'].fillna(0, inplace=True)
    
    # Calculate submission delay (date_submitted - date)
    assessment_merged['submission_delay'] = assessment_merged['date_submitted'] - assessment_merged['date']
    
    assessment_agg = assessment_merged.groupby('id_student').agg(
        assessments_attempted=('id_assessment', 'count'),
        average_assessment_score=('score', 'mean'),
        avg_submission_delay=('submission_delay', 'mean')
    ).reset_index()
    
    # Fill NaN delays with 0
    assessment_agg['avg_submission_delay'].fillna(0, inplace=True)
    
    # 4. Registration Feature Engineering
    # Handle missing unregistration dates (means they didn't withdraw)
    student_registration['date_unregistration'].fillna(999, inplace=True) 
    # Calculate registration duration before course start
    student_registration['registration_delay'] = student_registration['date_registration']
    student_registration['registration_delay'].fillna(0, inplace=True)
    
    reg_agg = student_registration.groupby('id_student').agg(
        avg_registration_delay=('registration_delay', 'mean')
    ).reset_index()

    # 5. Merge all aggregated data with student info
    df = pd.merge(student_info, vle_agg, on='id_student', how='left')
    df = pd.merge(df, assessment_agg, on='id_student', how='left')
    df = pd.merge(df, reg_agg, on='id_student', how='left')
    
    # 6. Handle Missing Values Post-Merge (Students with no VLE or Assessment data)
    fill_cols = [
        'total_vle_clicks', 'active_learning_days', 'avg_clicks_per_active_day',
        'assessments_attempted', 'average_assessment_score', 'avg_submission_delay',
        'avg_registration_delay'
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(0) # Fill with 0 for students who didn't interact
        
    print(f"Final Dataset Shape: {df.shape}")
    return df

def generate_early_warning_data(student_info, student_assessment, assessments, student_vle, cutoff_days=30):
    """
    Creates a dataset using only data available up to a certain number of days (e.g., first 30 days).
    """
    print(f"Generating early warning dataset (Cutoff: {cutoff_days} days)...")
    
    # Filter VLE interactions before cutoff
    early_vle = student_vle[student_vle['date'] <= cutoff_days]
    
    vle_agg = early_vle.groupby('id_student').agg(
        early_vle_clicks=('sum_click', 'sum'),
        early_active_days=('date', 'nunique')
    ).reset_index()
    
    # Filter Assessments submitted before cutoff
    early_assess = student_assessment[student_assessment['date_submitted'] <= cutoff_days]
    assess_merged = pd.merge(early_assess, assessments, on='id_assessment', how='left')
    
    assess_agg = assess_merged.groupby('id_student').agg(
        early_assessments_attempted=('id_assessment', 'count'),
        early_average_score=('score', 'mean')
    ).reset_index()
    
    # Merge
    df_early = pd.merge(student_info, vle_agg, on='id_student', how='left')
    df_early = pd.merge(df_early, assess_agg, on='id_student', how='left')
    
    fill_cols = ['early_vle_clicks', 'early_active_days', 'early_assessments_attempted', 'early_average_score']
    for col in fill_cols:
        df_early[col] = df_early[col].fillna(0)
        
    return df_early

def save_processed_data(df, output_path="data/processed/student_features.csv"):
    """Saves the processed dataframe to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Example execution flow
    info, assessment, assessments_df, vle, registration = load_data()
    if info is not None:
        processed_df = preprocess_and_engineer_features(info, assessment, assessments_df, vle, registration)
        save_processed_data(processed_df)
        
        early_df = generate_early_warning_data(info, assessment, assessments_df, vle, cutoff_days=30)
        save_processed_data(early_df, "data/processed/early_warning_features.csv")
