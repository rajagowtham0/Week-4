from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import embedding
import similarityengine
import Insight_generator
import main

app = FastAPI(title="CCMS AI Clinical Similarity API")

# Global variables
stored_cases = None
stored_embeddings = None
# startup event
@app.on_event("startup")
def startup_event():
    global stored_cases, stored_embeddings
    print("Initializing Similarity Engine...")
    stored_cases, stored_embeddings = main.initialize_system()
    print("System Ready!")

# request model
class CaseInput(BaseModel):
    symptoms: str
    doctor_notes: str

# health check
@app.get("/")
def health_check():
    return {"message": "CCMS API running"}

# main end point
@app.post("/analyze-case")
def analyze_case(case: CaseInput):

    try:
        input_text = f"{case.symptoms} {case.doctor_notes}"

        # Generate query embedding
        query_embedding = embedding.generate_embeddings([input_text])

        # Use similarity engine properly
        similar_cases = similarityengine.retrieve_similar_cases(
            query_embedding,
            stored_cases,
            stored_embeddings
        )

        insights = Insight_generator.generate_insights(similar_cases)

        return {
            "input_summary": input_text,
            "similar_cases": similar_cases,
            "clinical_insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))