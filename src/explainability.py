import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class ModelExplainer:
    def __init__(self, model_pipeline, feature_names):
        """
        Initializes with a trained sklearn pipeline. 
        Note: SHAP typically requires the base model and preprocessed data, 
        so we extract the classifier and preprocessor from the pipeline.
        """
        self.pipeline = model_pipeline
        self.classifier = model_pipeline.named_steps['classifier']
        self.preprocessor = model_pipeline.named_steps['preprocessor']
        self.feature_names = feature_names
        
        # We need to determine the exact feature names after one-hot encoding
        # For simplicity in this assignment, assuming tree-based models (RandomForest, XGBoost)
        # where SHAP TreeExplainer can be used.
        self.explainer = None
        self.setup_explainer()

    def clean_feature_names(self, names):
        cleaned = []
        for name in names:
            # Remove pipeline prefixes like num__ and cat__
            name = name.replace('num__', '').replace('cat__', '')
            # Replace underscores with spaces and capitalize
            name = name.replace('_', ' ').title()
            cleaned.append(name)
        return np.array(cleaned)

    def get_transformed_feature_names(self):
        """Helper to get feature names after pipeline preprocessing."""
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            return self.clean_feature_names(self.preprocessor.get_feature_names_out())
        return self.clean_feature_names(self.feature_names)

    def setup_explainer(self):
        # We use TreeExplainer for Random Forest and XGBoost.
        # For Logistic Regression we would use LinearExplainer, but TreeExplainer is best for ensemble methods.
        try:
            self.explainer = shap.TreeExplainer(self.classifier)
        except Exception as e:
            print(f"Could not initialize TreeExplainer: {e}. Defaulting to Explainer.")
            self.explainer = shap.Explainer(self.classifier)

    def generate_global_explanation(self, X_sample, output_path="screenshots/global_shap.png"):
        """Generates and saves a SHAP summary plot for global feature importance."""
        print("Generating global explanation...")
        X_transformed = self.preprocessor.transform(X_sample)
        
        # In case it returns sparse matrix, convert to dense
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
            
        shap_values = self.explainer.shap_values(X_transformed)
        
        # Handle binary classification shap_values shape (some versions of SHAP return list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Use values for positive class (At Risk)
            
        feature_names = self.get_transformed_feature_names()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Global explanation saved to {output_path}")

    def generate_local_explanation(self, X_instance, instance_idx=0, output_path="screenshots/local_shap.png"):
        """Generates and saves a SHAP waterfall plot for a single prediction."""
        X_transformed = self.preprocessor.transform(X_instance)
        
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
            
        shap_values = self.explainer(X_transformed)
        
        feature_names = self.get_transformed_feature_names()
        
        # Extract specific instance
        sv_instance = shap_values[instance_idx]
        
        # For binary classification, pick class 1
        if len(sv_instance.values.shape) > 1:
            sv_instance.values = sv_instance.values[:, 1]
            sv_instance.base_values = sv_instance.base_values[1]
            
        sv_instance.feature_names = feature_names
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(sv_instance, show=False)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        # Textual local explanation (Human-readable)
        sorted_indices = np.argsort(np.abs(sv_instance.values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:3]]
        
        explanation = f"🔍 **Easy Explanation:** The AI analyzed the student's profile and found that their **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}** were the biggest reasons for this prediction."
        return explanation

if __name__ == "__main__":
    print("Explainability module ready.")
