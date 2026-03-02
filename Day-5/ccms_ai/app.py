from fastapi import FastAPI, HTTPException
import numpy as np
import time

from models import CaseRequest, CaseResponse
from embedding import combine_text, generate_embedding
from similarityengine import retrieve_similar_cases
from Insight_generator import generate_case_insight
from database import fetch_all_cases

app = FastAPI(title="CCMS AI Similarity Engine")
# global in memory storage and cache key
stored_cases = []
stored_embeddings = None
cache = {}
# start-up event
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

    global stored_cases, stored_embeddings, cache

    start_time = time.time()

    try:
        # runtime validation
        if not request.symptoms.strip() or not request.doctor_notes.strip():
            raise HTTPException(
                status_code=400,
                detail="Symptoms and doctor notes cannot be empty."
            )

        if not stored_cases or stored_embeddings is None:
            raise HTTPException(
                status_code=500,
                detail="System not initialized properly."
            )

        combined_text = combine_text(
            request.symptoms,
            request.doctor_notes
        )

        cache_key = combined_text.strip().lower()

        if cache_key in cache:
            return cache[cache_key]

        query_embedding = generate_embedding(combined_text)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similar_cases = retrieve_similar_cases(
            query_embedding=query_embedding,
            stored_cases=stored_cases,
            stored_embeddings=stored_embeddings
        )

        insight_summary, confidence_reason = generate_case_insight(
            similar_cases,
            stored_cases
        )

        # Calculate output quality (same as confidence score)
        similarity_scores = [case["similarity_score"] for case in similar_cases]
        output_quality = round(
            sum(similarity_scores) / len(similarity_scores), 4
        ) if similarity_scores else 0.0

        end_time = time.time()
        response_time = round(end_time - start_time, 4)

        # performance metrics
        print("\n Performance Metrics)
        print("Response Time (seconds):", response_time)
        print("Output Quality:", output_quality)

        response = {
            "similar_cases": similar_cases,
            "insight_summary": insight_summary,
            "confidence_reason": confidence_reason
        }

        cache[cache_key] = response

        return response

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )