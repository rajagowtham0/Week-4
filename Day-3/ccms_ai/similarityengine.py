import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# retrieving similar cases
def retrieve_similar_cases(query_embedding, stored_cases, stored_embeddings, top_k=4):
    # safety checks
    if query_embedding is None:
        raise ValueError("Query embedding is None.")
    if stored_embeddings is None or len(stored_embeddings) == 0:
        raise ValueError("Stored embeddings are empty.")
    if not stored_cases:
        return []
    # Ensure query_embedding shape is correct
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # cosine similarity calculations
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    # response construct
    for idx in top_indices:

        # Safety check for index bounds
        if idx >= len(stored_cases):
            continue

        results.append({
            "case_id": stored_cases[idx]["case_id"], 
            "similarity_score": round(float(similarities[idx]), 4)
        })

    return results