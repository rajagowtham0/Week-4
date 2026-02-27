from fastapi import FastAPI, HTTPException
import numpy as np

from models import CaseRequest, CaseResponse
from embedding import combine_text, generate_embedding
from similarityengine import retrieve_similar_cases
from Insight_generator import generate_case_insight
from database import fetch_all_cases

app = FastAPI(title="CCMS AI Similarity Engine")
# global in-memory storage
stored_cases = []
stored_embeddings = None
# startup event
@app.on_event("startup")
def load_data():
    global stored_cases, stored_embeddings

    stored_cases = fetch_all_cases()

    if not stored_cases:
        raise RuntimeError("No records found in MongoDB.")

    stored_embeddings = np.array([
        generate_embedding(
            combine_text(case["symptoms"], case["doctor_notes"])
        )
        for case in stored_cases
    ])

# main endpoint
@app.post("/analyze-case", response_model=CaseResponse)
def analyze_case(request: CaseRequest):

    global stored_cases, stored_embeddings

    try:
        if not stored_cases or stored_embeddings is None:
            raise HTTPException(
                status_code=500,
                detail="System not initialized properly."
            )

        # Generate embedding for incoming case
        combined_text = combine_text(
            request.symptoms,
            request.doctor_notes
        )

        query_embedding = generate_embedding(combined_text)

        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Retrieve similar cases
        similar_cases = retrieve_similar_cases(
            query_embedding=query_embedding,
            stored_cases=stored_cases,
            stored_embeddings=stored_embeddings
        )

        # Generate insight
        insight_summary, confidence_reason = generate_case_insight(
            similar_cases,
            stored_cases
        )

        return {
            "similar_cases": similar_cases,
            "insight_summary": insight_summary,
            "confidence_reason": confidence_reason
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )