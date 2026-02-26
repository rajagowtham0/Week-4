from collections import Counter


def generate_case_insight(similar_cases, stored_cases):

    if not similar_cases:
        return (
            "No treatment pattern identified.",
            "Limited historical similarity."
        )

    case_map = {case["case_id"]: case for case in stored_cases}

    treatments = []
    similarity_scores = []

    for case in similar_cases:
        case_id = case["case_id"]
        similarity_scores.append(case["similarity_score"])

        matched_case = case_map.get(case_id)

        if matched_case and "treatment" in matched_case:
            treatments.append(matched_case["treatment"])

    if not treatments:
        return (
            "No treatment pattern identified.",
            "Limited historical similarity."
        )

    most_common = Counter(treatments).most_common(1)[0][0]

    summary = (
        f"In similar past cases, patients commonly responded well to {most_common}."
    )

    # Calculate average similarity score
    avg_confidence = round(
        sum(similarity_scores) / len(similarity_scores),
        4
    )

    confidence = (
        f"Based on {len(similar_cases)} similar historical cases, "
        f"the confidence score obtained is {avg_confidence}."
    )

    return summary, confidence