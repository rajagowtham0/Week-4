import pandas as pd
import embedding
import similarityengine
# Global variables
stored_cases = None
stored_embeddings = None

def initialize_system():

    global stored_cases, stored_embeddings

    # Load dataset
    df = pd.read_csv("cases.csv")

    # Combine text
    df["combined_text"] = df["symptoms"] + " " + df["doctor_notes"]

    # Generate embeddings
    embeddings = embedding.generate_embeddings(df["combined_text"].tolist())

    stored_cases = df
    stored_embeddings = embeddings

    return stored_cases, stored_embeddings