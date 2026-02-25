from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import embedding
import similarityengine
import Insight_generator
import main

app = FastAPI(
    title="CCMS AI Clinical Similarity API",
    version="4.0.0"
)

stored_cases = None
stored_embeddings = None
# loading of dataset
@app.on_event("startup")
def startup_event():
    global stored_cases, stored_embeddings
    stored_cases, stored_embeddings = main.initialize_system()
# input schema
class CaseInput(BaseModel):
    symptoms: str
    doctor_notes: str
# main endpoint
@app.post("/analyze-case")
def analyze_case(case: CaseInput):

    try:
        input_text = f"{case.symptoms} {case.doctor_notes}"

        query_embedding = embedding.generate_embeddings([input_text])

        similar_cases = similarityengine.retrieve_similar_cases(
            query_embedding=query_embedding,
            stored_cases=stored_cases,
            stored_embeddings=stored_embeddings,
            top_k=4
        )

        insights = Insight_generator.generate_insights(
            similar_cases=similar_cases,
            stored_cases=stored_cases
        )

        # Final required structured output
        return {
            "similar_cases": similar_cases,
            **insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))