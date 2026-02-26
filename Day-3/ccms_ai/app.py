from fastapi import FastAPI, HTTPException
from models import CaseRequest, CaseResponse
from embedding import combine_text, generate_embedding
from similarity_engine import retrieve_similar_cases
from insight_generator import generate_case_insight
from database import fetch_all_cases

app = FastAPI(title="CCMS AI Similarity Engine")


@app.post("/analyze-case", response_model=CaseResponse)
def analyze_case(request: CaseRequest):

    try:
        # Combine input text
        combined_text = combine_text(
            request.symptoms,
            request.doctor_notes
        )

        # Generate embedding for incoming case
        query_embedding = generate_embedding(combined_text)

        # Fetch stored cases from MongoDB
        stored_cases = fetch_all_cases()

        if not stored_cases:
            raise HTTPException(
                status_code=404,
                detail="No patient records found in database."
            )

        # Generate embeddings for stored cases
        stored_embeddings = [
            generate_embedding(
                combine_text(case["symptoms"], case["doctor_notes"])
            )
            for case in stored_cases
        ]

        # Retrieve similar cases
        similar_cases = retrieve_similar_cases(
            query_embedding=query_embedding,
            stored_cases=stored_cases,
            stored_embeddings=stored_embeddings
        )

        # Generate insight summary
        insight_summary, confidence_reason = generate_case_insight(
            similar_cases,
            stored_cases
        )

        return {
            "similar_cases": similar_cases,
            "insight_summary": insight_summary,
            "confidence_reason": confidence_reason
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )