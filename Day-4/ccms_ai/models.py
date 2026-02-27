from pydantic import BaseModel
from typing import List

# request schema
class CaseRequest(BaseModel):
    symptoms: str
    doctor_notes: str

# similar case schema
class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float

# response schema
class CaseResponse(BaseModel):
    similar_cases: List[SimilarCase]
    insight_summary: str
    confidence_reason: str