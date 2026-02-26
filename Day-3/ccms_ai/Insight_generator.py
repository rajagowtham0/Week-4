# insight_generator.py

from collections import Counter

#generating structured insight summary
def generate_case_insight(similar_cases, stored_cases):
    # safety checks
    if not similar_cases:
        return (
            "No treatment pattern identified.",
            "Limited historical similarity."
        )

    if not stored_cases:
        return (
            "No treatment pattern identified.",
            "No historical data available."
        )

    # Create quick lookup dictionary
    case_map = {
        case["case_id"]: case for case in stored_cases
    }

    treatments = []
    
    # extracting treatment information
    for case in similar_cases:
        case_id = case.get("case_id")
        matched_case = case_map.get(case_id)

        if matched_case and "treatment" in matched_case:
            treatments.append(matched_case["treatment"])

    # structured insight summary
    if not treatments:
        return (
            "No treatment pattern identified.",
            "Limited historical similarity."
        )

    most_common = Counter(treatments).most_common(1)[0][0]

    summary = (
        f"In similar past cases, patients commonly responded well to {most_common}."
    )

    confidence = (
        f"Based on {len(similar_cases)} similar historical cases."
    )

    return summary, confidence