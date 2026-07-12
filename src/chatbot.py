import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import re
import torch

class UniversityChatbot:
    def __init__(self, faq_path="data/university_faq.csv", model_name='all-MiniLM-L6-v2'):
        self.faq_path = faq_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.faq_data = None
        self.question_embeddings = None
        self.load_data_and_encode()

    def clean_text(self, text):
        """Basic text cleaning."""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def load_data_and_encode(self):
        """Loads FAQ data and pre-computes embeddings for fast retrieval."""
        try:
            self.faq_data = pd.read_csv(self.faq_path)
            
            # Combine question and keywords for better context
            self.faq_data['search_text'] = self.faq_data['question'] + " " + self.faq_data['keywords'].fillna('')
            
            # Compute embeddings
            print("Encoding FAQ questions...")
            self.question_embeddings = self.model.encode(
                self.faq_data['search_text'].tolist(), convert_to_tensor=True
            )
            print("Chatbot ready.")
        except FileNotFoundError:
            print(f"Error: Could not find {self.faq_path}. Please run generate_datasets.py first.")

    def get_answer(self, user_question, confidence_threshold=0.5):
        """
        Takes a natural language question and returns the most relevant answer.
        """
        if self.faq_data is None:
            return "Chatbot is not initialized properly. Data missing.", 0.0, None

        # Clean and encode user question
        cleaned_question = self.clean_text(user_question)
        question_embedding = self.model.encode(cleaned_question, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(question_embedding, self.question_embeddings)[0]
        
        # Find best match
        best_match_idx = torch.argmax(cosine_scores).item() if 'torch' in str(type(cosine_scores)) else np.argmax(cosine_scores.cpu().numpy())
        score = cosine_scores[best_match_idx].item()

        if score >= confidence_threshold:
            best_row = self.faq_data.iloc[best_match_idx]
            return best_row['answer'], score, best_row['category']
        else:
            # Fallback mechanism
            self.log_unanswered(user_question)
            return "I could not find a sufficiently reliable answer. Please contact the relevant university office or rephrase your question.", score, "Fallback"
            
    def log_unanswered(self, question, log_file="data/unanswered_questions.txt"):
        """Logs questions that fall below the confidence threshold."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(question + "\n")

if __name__ == "__main__":
    # Test cases
    chatbot = UniversityChatbot()
    test_questions = [
        "How can I register for a course?",
        "What attendance percentage is required?",
        "Can my tuition fee be paid in instalments?",
        "Tell me tomorrow's weather."
    ]
    
    for q in test_questions:
        print(f"\nUser: {q}")
        answer, score, category = chatbot.get_answer(q)
        print(f"Bot (Score: {score:.2f}, Category: {category}): {answer}")
