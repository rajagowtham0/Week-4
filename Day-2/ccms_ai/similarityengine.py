import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# similarity engine
def retrieve_similar_cases(query_embedding, stored_cases, stored_embeddings, top_k=4):

    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "case_id": stored_cases.iloc[idx]["case_id"],
            "similarity_score": round(float(similarities[idx]), 4)
        })

    return results