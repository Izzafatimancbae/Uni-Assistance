import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def prepare_modeling_data(df_path="data/processed/student_features.csv"):
    """Loads and splits the dataset for modeling."""
    print(f"Loading data from {df_path}")
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        print(f"Error: {df_path} not found. Please run preprocessing.py first.")
        return None, None, None, None, None, None

    # Drop identifying columns and target from features
    target_col = 'At_Risk'
    drop_cols = ['id_student', 'code_module', 'code_presentation', 'final_result', target_col]
    drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, categorical_features, numeric_features

def build_pipeline(classifier, categorical_features, numeric_features):
    """Builds a scikit-learn pipeline with preprocessing and the specified classifier."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluates the model and returns a metrics dictionary."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_test))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if np.any(y_proba) else None

    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc: print(f"ROC-AUC:   {roc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not At Risk', 'At Risk']))

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc
    }

def run_cross_validation(pipeline, X_train, y_train, model_name, cv_folds=5):
    """Runs stratified k-fold cross-validation and prints results."""
    print(f"\n  Running {cv_folds}-Fold Cross-Validation for {model_name}...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"  CV F1 Scores: {cv_scores.round(4)}")
    print(f"  Mean CV F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return cv_scores.mean()

def hyperparameter_tuning(base_clf, cat_features, num_features, X_train, y_train, model_name):
    """Performs GridSearchCV hyperparameter tuning for a given classifier."""
    print(f"\n  Tuning hyperparameters for {model_name}...")

    pipeline = build_pipeline(base_clf, cat_features, num_features)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if model_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.05, 0.1]
        }
    else:
        return None

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='f1',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best Params: {grid_search.best_params_}")
    print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate(is_early_warning=False):
    """Trains Logistic Regression, Random Forest (tuned), and XGBoost (tuned)."""
    path = "data/processed/early_warning_features.csv" if is_early_warning else "data/processed/student_features.csv"
    data = prepare_modeling_data(path)

    if data[0] is None:
        return

    X_train, X_test, y_train, y_test, cat_features, num_features = data

    # Calculate class weight ratio for XGBoost
    scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos)
    }

    results = {}
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None

    # Step 1: Train all base models and run cross-validation
    fitted_pipelines = {}
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(clf, cat_features, num_features)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        run_cross_validation(pipeline, X_train, y_train, name)

    # Step 2: Hyperparameter tuning for RF and XGBoost
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    tuned_rf = hyperparameter_tuning(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        cat_features, num_features, X_train, y_train, "Random Forest"
    )
    tuned_xgb = hyperparameter_tuning(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos),
        cat_features, num_features, X_train, y_train, "XGBoost"
    )

    if tuned_rf: fitted_pipelines["Random Forest (Tuned)"] = tuned_rf
    if tuned_xgb: fitted_pipelines["XGBoost (Tuned)"] = tuned_xgb

    # Step 3: Evaluate all models
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    for name, pipeline in fitted_pipelines.items():
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = metrics

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")

    # Save results and best model
    os.makedirs('models', exist_ok=True)
    model_filename = 'models/early_risk_prediction_model.pkl' if is_early_warning else 'models/risk_prediction_model.pkl'
    results_filename = 'models/early_model_results.pkl' if is_early_warning else 'models/model_results.pkl'

    with open(model_filename, 'wb') as f:
        pickle.dump(best_pipeline, f)
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved best model to {model_filename}")
    print(f"Saved results to {results_filename}")
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("=== TRAINING FULL SEMESTER MODELS ===")
    print("=" * 60)
    full_results = train_and_evaluate(is_early_warning=False)

    print("\n" + "=" * 60)
    print("=== TRAINING EARLY WARNING MODELS (First 30 Days) ===")
    print("=" * 60)
    early_results = train_and_evaluate(is_early_warning=True)
