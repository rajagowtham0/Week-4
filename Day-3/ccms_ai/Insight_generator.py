from collections import Counter
import pandas as pd


def generate_insights(similar_cases, stored_cases):

    treatments = []
    outcomes = []
    similarity_scores = []

    for case in similar_cases:
        case_id = case["case_id"]
        similarity_scores.append(case["similarity_score"])

        row = stored_cases[stored_cases["case_id"] == case_id].iloc[0]

        treatments.append(str(row["treatment"]))
        outcomes.append(str(row["outcome"]))

        # Safe handling for recovery_days 
        recovery_value = row["recovery_days"]

        if pd.isna(recovery_value):
            recovery_value = None
        else:
            recovery_value = int(recovery_value)

    # Common treatment pattern
    common_treatment_pattern = list(set(treatments))

    # Most frequent outcome
    outcome_pattern = Counter(outcomes).most_common(1)[0][0]

    # Average similarity
    avg_confidence = round(sum(similarity_scores) / len(similarity_scores), 4)

    confidence_reason = (
        f"Based on {len(similar_cases)} highly similar historical cases, "
        f"the confidence score obtained is {avg_confidence}."
    )

    return {
        "similar_cases": similar_cases,
        "common_treatment_pattern": common_treatment_pattern,
        "outcome_pattern": outcome_pattern,
        "confidence_reason": confidence_reason
    }