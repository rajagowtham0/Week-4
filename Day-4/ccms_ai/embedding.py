# functional embedding generation module
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
# Private module-level variable
_model = None
# loading of model
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model
# generate embedding for multiple texts 
def generate_embeddings(texts: List[str]) -> np.ndarray:
    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")

    if len(texts) == 0:
        raise ValueError("Text list is empty.")

    model = load_embedding_model()

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False   # turned off for API performance
    )
# generate embedding for single terms (symptoms)
def generate_single_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    model = load_embedding_model()

    return model.encode(
        text,
        convert_to_numpy=True
    )
# wrapper function to maintain compatibility
def generate_embedding(text: str) -> np.ndarray:
    return generate_single_embedding(text)
# combining the text utility
def combine_text(symptoms: str, doctor_notes: str) -> str:
    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Both symptoms and doctor_notes must be strings.")
    return f"{symptoms}. {doctor_notes}"