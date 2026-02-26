from fastapi import FastAPI, HTTPException
import numpy as np

from models import CaseRequest, CaseResponse
from embedding import combine_text, generate_embedding
from similarityengine import retrieve_similar_cases
from Insight_generator import generate_case_insight
from database import fetch_all_cases

app = FastAPI(title="CCMS AI Similarity Engine")


@app.post("/analyze-case", response_model=CaseResponse)
def analyze_case(request: CaseRequest):

    try:
        # Step 1: Combine input text
        combined_text = combine_text(
            request.symptoms,
            request.doctor_notes
        )

        # Step 2: Generate embedding for incoming case
        query_embedding = generate_embedding(combined_text)

        # Ensure correct shape for cosine similarity
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Step 3: Fetch stored cases from MongoDB
        stored_cases = fetch_all_cases()

        if not stored_cases:
            raise HTTPException(
                status_code=404,
                detail="No patient records found in database."
            )

        # Step 4: Generate embeddings for stored cases
        stored_embeddings = np.array([
            generate_embedding(
                combine_text(case["symptoms"], case["doctor_notes"])
            )
            for case in stored_cases
        ])

        # Step 5: Retrieve similar cases
        similar_cases = retrieve_similar_cases(
            query_embedding=query_embedding,
            stored_cases=stored_cases,
            stored_embeddings=stored_embeddings
        )

        # Step 6: Generate insight summary
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