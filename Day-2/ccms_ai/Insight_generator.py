# functional insight generator
from collections import Counter
import re

# similarity case insight generation
def generate_case_insight(similar_cases: list,
                          stored_cases: list,
                          query_text: str) -> dict:

    if len(similar_cases) == 0:
        return {
            "shared_terms": [],
            "common_treatments": [],
            "outcome_trend": "Insufficient data",
            "recovery_trend": "Insufficient data"
        }

    # Map case_id to case data
    case_map = {
        case["case_id"]: case for case in stored_cases
    }

    similar_texts = []
    treatments = []
    outcomes = []
    recovery_info = []

    for case in similar_cases:
        case_data = case_map.get(case["case_id"])

        if not case_data:
            continue

        similar_texts.append(case_data.get("combined_text", ""))

        if "treatment" in case_data:
            treatments.append(case_data["treatment"])

        if "outcome" in case_data:
            outcomes.append(case_data["outcome"])

        if "recovery_time" in case_data:
            recovery_info.append(case_data["recovery_time"])

    # Shared Terms 
    query_words = set(query_text.lower().split())
    term_counter = Counter()

    for text in similar_texts:
        words = set(text.lower().split())
        shared = query_words.intersection(words)
        term_counter.update(shared)

    most_common_terms = [
        term for term, _ in term_counter.most_common(5)
    ]

    #  Common Treatments 
    most_common_treatments = [
        t for t, _ in Counter(treatments).most_common(3)
    ]

    # Outcome Trend 
    outcome_trend = "Not clearly observed"
    if outcomes:
        outcome_trend = Counter(outcomes).most_common(1)[0][0]

    # Recovery Trend 
    recovery_trend = "Not specified"
    if recovery_info:
        recovery_trend = "Recovery mentioned in similar cases"

    insight = {
        "shared_terms": most_common_terms,
        "common_treatments": most_common_treatments,
        "outcome_trend": outcome_trend,
        "recovery_trend": recovery_trend
    }

    return insight