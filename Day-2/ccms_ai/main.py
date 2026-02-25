import pandas as pd
import embedding

stored_cases = None
stored_embeddings = None


def initialize_system():

    global stored_cases, stored_embeddings

    # Load dataset
    df = pd.read_csv(r"C:\Users\rajak\Downloads\Week_0_Prep_Week_Ssample Data_clinic_cases.csv")

    # Combine text fields
    df["combined_text"] = df["symptoms"] + " " + df["doctor_notes"]

    # Generate embeddings for stored cases
    embeddings = embedding.generate_embeddings(df["combined_text"].tolist())

    stored_cases = df
    stored_embeddings = embeddings

    return stored_cases, stored_embeddings