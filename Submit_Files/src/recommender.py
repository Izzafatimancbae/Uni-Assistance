import pandas as pd
import numpy as np

class LearningRecommender:
    def __init__(self, resources_path="data/learning_resources.csv"):
        self.resources_path = resources_path
        self.resources = None
        self.load_resources()

    def load_resources(self):
        try:
            self.resources = pd.read_csv(self.resources_path)
            print("Learning resources loaded.")
        except FileNotFoundError:
            print(f"Error: Could not find {self.resources_path}.")

    def get_recommendation(self, student_data):
        """
        Takes student dictionary/Series and recommends a resource based on rules.
        Expected student_data keys:
        - Risk_Level (High/Medium/Low)
        - average_assessment_score
        - avg_clicks_per_active_day
        """
        if self.resources is None or self.resources.empty:
            return None

        risk = student_data.get('Risk_Level', 'Low')
        score = student_data.get('average_assessment_score', 100)
        engagement = student_data.get('avg_clicks_per_active_day', 10)

        weakness = ""
        recommended_difficulty = "Beginner"
        reason = ""

        # Rule 1: High risk with low assessment score -> Beginner Tutorials/Videos
        if risk == 'High' and score < 50:
            weakness = "Low assessment performance"
            recommended_difficulty = "Beginner"
            reason = "The student's average assessment score is below 50% and is at high academic risk."
            
        # Rule 2: Medium risk with low engagement -> Short Interactive/Video
        elif risk in ['High', 'Medium'] and engagement < 5:
            weakness = "Low platform engagement"
            recommended_difficulty = "Beginner"
            reason = "The student recorded very few learning interactions compared to expectations."
            
        # Rule 3: Medium/Low risk with decent score but struggling with advanced topics
        elif risk == 'Medium' and score >= 50 and score < 70:
            weakness = "Requires practice"
            recommended_difficulty = "Intermediate"
            reason = "The student is passing but could improve scores with targeted practice exercises."
            
        # Rule 4: Low risk with high performance -> Advanced Readings
        elif risk == 'Low' and score >= 70:
            weakness = "Ready for advanced material"
            recommended_difficulty = "Advanced"
            reason = "The student achieved consistently high assessment scores."
            
        else:
            weakness = "General review needed"
            recommended_difficulty = "Intermediate"
            reason = "General recommendation based on average profile."

        # Filter resources by difficulty and module
        student_module = student_data.get('code_module')
        
        suitable_resources = self.resources[self.resources['difficulty'] == recommended_difficulty]
        
        if student_module:
            module_resources = suitable_resources[suitable_resources['module'] == student_module]
            if not module_resources.empty:
                suitable_resources = module_resources
        
        if not suitable_resources.empty:
            chosen = suitable_resources.sample(n=1).iloc[0]
            
            return {
                "Student Risk Level": risk,
                "Identified Need": weakness,
                "Recommended Resource": chosen['title'],
                "Resource Type": chosen['resource_type'],
                "Difficulty Level": chosen['difficulty'],
                "Explanation": reason,
                "URL": chosen['url']
            }
        return None

if __name__ == "__main__":
    recommender = LearningRecommender()
    
    test_cases = [
        {"Risk_Level": "High", "average_assessment_score": 40, "avg_clicks_per_active_day": 8},
        {"Risk_Level": "Medium", "average_assessment_score": 65, "avg_clicks_per_active_day": 3},
        {"Risk_Level": "Low", "average_assessment_score": 90, "avg_clicks_per_active_day": 15}
    ]
    
    for idx, tc in enumerate(test_cases):
        print(f"\n--- Test Case {idx+1} ---")
        rec = recommender.get_recommendation(tc)
        for k, v in rec.items():
            print(f"{k}: {v}")
