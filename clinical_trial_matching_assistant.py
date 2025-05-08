
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_trials(file_path):
    logging.info("Loading clinical trial data...")
    df = pd.read_csv(file_path)
    return df

def embed_text(text_list):
    return embedding_model.encode(text_list, convert_to_tensor=False)

def match_trials(patient_description, trials_df, top_k=3):
    logging.info("Embedding and matching patient to trials...")
    trial_texts = trials_df['eligibility_criteria'].fillna('').tolist()
    trial_embeddings = embed_text(trial_texts)
    patient_embedding = embed_text([patient_description])[0]

    similarities = cosine_similarity([patient_embedding], trial_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    matched_trials = trials_df.iloc[top_indices].copy()
    matched_trials['similarity_score'] = similarities[top_indices]
    return matched_trials

def explain_match(patient_description, trial_summary):
    prompt = f"""
    A patient has the following medical background: {patient_description}

    Given the clinical trial summary: {trial_summary}

    Explain in detail why this trial might be a good match for the patient, including potential benefits and fit with eligibility criteria.
    """

    logging.info("Generating match explanation using GPT...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a clinical research assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

if __name__ == "__main__":
    trials_file = "clinical_trials.csv"  # Must include 'eligibility_criteria' column
    trials_df = load_trials(trials_file)

    patient_input = input("Enter patient medical summary (conditions, age, medications, etc.): ")
    matched = match_trials(patient_input, trials_df)

    print("\nTop Matching Trials:\n")
    for idx, row in matched.iterrows():
        print(f"Trial Title: {row['title']}")
        print(f"Eligibility: {row['eligibility_criteria'][:300]}...")
        explanation = explain_match(patient_input, row['eligibility_criteria'])
        print(f"GPT Explanation: {explanation}\n{'-'*60}\n")
