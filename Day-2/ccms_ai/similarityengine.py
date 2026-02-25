# functional similarity engine module
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# similarity engine module
def retrieve_similar_cases(new_embedding,
                           stored_cases,
                           top_n=4):

    stored_embeddings = np.array(
        [case["embedding"] for case in stored_cases]
    )

    if new_embedding.ndim == 1:
        new_embedding = new_embedding.reshape(1, -1)

    similarity_scores = cosine_similarity(
        new_embedding,
        stored_embeddings
    )[0]

    ranked_indices = np.argsort(similarity_scores)[::-1][:top_n]

    results = []

    for idx in ranked_indices:
        case = stored_cases[idx]
        results.append({
            "case_id": case["case_id"],
            "similarity_score": float(similarity_scores[idx]),
            "treatment": case.get("treatment", "Not Available"),
            "recovery_period": case.get("recovery_period", "Not Available")
        })

    return results