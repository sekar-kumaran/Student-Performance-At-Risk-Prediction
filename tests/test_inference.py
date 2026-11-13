"""
Unit and integration tests for real-time model inference, probability bounds,
directional monotonicity, explainability payloads, and prediction determinism.
"""

from __future__ import annotations
from typing import Any, Dict, List
import pytest
from src.prediction import prepare_student_features, predict_risk


#
# 1. Live Prediction Execution (No Hardcoding)
def test_live_prediction_not_hardcoded(
    trained_model,
    feature_order: List[str],
    low_risk_student_input: Dict[str, float],
    high_risk_student_input: Dict[str, float],
    metadata: Dict[str, Any],
):
    """
    Verify predictions are dynamically computed by the trained model pipeline,
    producing strictly different probabilities and classification labels.
    """
    threshold = metadata.get("classification_threshold", 0.67)

    df_low = prepare_student_features(low_risk_student_input, feature_order)
    result_low = predict_risk(trained_model, df_low, threshold=threshold)

    df_high = prepare_student_features(high_risk_student_input, feature_order)
    result_high = predict_risk(trained_model, df_high, threshold=threshold)

    # Probabilities must be strictly different
    assert result_low["probability"] != result_high["probability"], (
        "Model returned identical probability for contrasting student profiles!"
    )

    # Classifications must differentiate high-risk from low-risk
    assert result_low["is_at_risk"] is False
    assert result_low["label"] == "Low Risk"
    assert result_high["is_at_risk"] is True
    assert result_high["label"] == "At-Risk"


# 2. Probability Bounds Validation
def test_prediction_probabilities_strictly_bounded(
    trained_model,
    feature_order: List[str],
    low_risk_student_input: Dict[str, float],
    high_risk_student_input: Dict[str, float],
    borderline_student_input: Dict[str, float],
    edge_case_inputs: Dict[str, Any],
):
    """
    Verify that predicted probabilities are valid floats strictly within [0.0, 1.0]
    across standard, borderline, and boundary profiles.
    """
    test_cases = [
        low_risk_student_input,
        high_risk_student_input,
        borderline_student_input,
        edge_case_inputs["all_zero"],
        edge_case_inputs["all_hundred"],
        edge_case_inputs["empty_dict"],
    ]

    for case in test_cases:
        df = prepare_student_features(case, feature_order)
        result = predict_risk(trained_model, df)

        proba = result["probability"]
        assert isinstance(proba, float), f"Probability is not a float: {type(proba)}"
        assert 0.0 <= proba <= 1.0, f"Probability {proba} is outside [0.0, 1.0]"


# 3. Directional Monotonicity (Domain Sanity Check)
def test_directional_monotonicity_risk_ordering(
    trained_model,
    feature_order: List[str],
    low_risk_student_input: Dict[str, float],
    borderline_student_input: Dict[str, float],
    high_risk_student_input: Dict[str, float],
):
    """
    DOMAIN SANITY CHECK: Assert risk probability strictly respects directional monotonicity:
    P(Risk | High-Risk Profile) > P(Risk | Borderline Profile) > P(Risk | Low-Risk Profile)
    """
    df_low = prepare_student_features(low_risk_student_input, feature_order)
    df_border = prepare_student_features(borderline_student_input, feature_order)
    df_high = prepare_student_features(high_risk_student_input, feature_order)

    p_low = predict_risk(trained_model, df_low)["probability"]
    p_border = predict_risk(trained_model, df_border)["probability"]
    p_high = predict_risk(trained_model, df_high)["probability"]

    assert p_high > p_border, f"High-risk prob ({p_high}) not greater than borderline ({p_border})"
    assert p_border > p_low, f"Borderline prob ({p_border}) not greater than low-risk ({p_low})"
    assert p_high > p_low, f"High-risk prob ({p_high}) not greater than low-risk ({p_low})"


# 4. Advisory Risk Band & Influential Signals Payload Schema
def test_prediction_payload_schema_and_types(
    trained_model,
    feature_order: List[str],
    high_risk_student_input: Dict[str, float],
):
    """
    Verify that the inference return payload contains all expected keys,
    risk categories, and actionable recommendations.
    """
    df = prepare_student_features(high_risk_student_input, feature_order)
    result = predict_risk(trained_model, df, threshold=0.67)

    required_keys = [
        "probability",
        "threshold",
        "is_at_risk",
        "label",
        "advisory_status",
        "recommendations",
        "derived_features",
    ]
    for key in required_keys:
        assert key in result, f"Inference payload missing required key: '{key}'"

    assert isinstance(result["probability"], float)
    assert isinstance(result["threshold"], float)
    assert isinstance(result["is_at_risk"], bool)
    assert result["label"] in ["At-Risk", "Low Risk"]
    assert result["advisory_status"] in [
        "Requires Additional Academic Support",
        "Satisfactory Academic Progress",
    ]
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0

    # Verify derived features sub-dictionary
    derived = result["derived_features"]
    assert isinstance(derived, dict)
    for derived_key in ["attendance_drop", "average_previous_score", "score_trend", "score_volatility"]:
        assert derived_key in derived, f"Missing derived feature '{derived_key}'"
        assert isinstance(derived[derived_key], float)


def test_actionable_recommendations_for_at_risk_students(
    trained_model,
    feature_order: List[str],
    high_risk_student_input: Dict[str, float],
):
    """
    Verify that an identified at-risk student receives targeted, actionable support recommendations
    rather than a generic black-box score.
    """
    df = prepare_student_features(high_risk_student_input, feature_order)
    result = predict_risk(trained_model, df)

    recs = result["recommendations"]
    assert len(recs) >= 2, "Expected multiple tailored recommendations for critical need student"

    joined_recs = " ".join(recs).lower()
    assert "tutoring" in joined_recs, "Missing academic tutoring recommendation"
    assert "assignment" in joined_recs, "Missing assignment follow-up recommendation"



# 5. Prediction Determinism & Reproducibility

def test_prediction_determinism(
    trained_model,
    feature_order: List[str],
    high_risk_student_input: Dict[str, float],
):
    """
    Verify prediction function is strictly deterministic:
    Identical inputs must yield bitwise-identical outputs on repeated execution.
    """
    df = prepare_student_features(high_risk_student_input, feature_order)

    run_1 = predict_risk(trained_model, df)
    run_2 = predict_risk(trained_model, df)

    assert run_1["probability"] == run_2["probability"]
    assert run_1["is_at_risk"] == run_2["is_at_risk"]
    assert run_1["label"] == run_2["label"]
    assert run_1["recommendations"] == run_2["recommendations"]
    assert run_1["derived_features"] == run_2["derived_features"]


# 6. Classification Threshold Sensitivity

def test_classification_threshold_sensitivity(
    trained_model,
    feature_order: List[str],
    high_risk_student_input: Dict[str, float],
):
    """
    Verify threshold flexibility: Adjusting the decision threshold changes
    the boolean classification without modifying the predicted raw probability.
    """
    df = prepare_student_features(high_risk_student_input, feature_order)

    # For high-risk student (proba ~0.85):
    # With threshold 0.50, student is flagged (0.85 >= 0.50)
    res_lower_thresh = predict_risk(trained_model, df, threshold=0.50)
    # With threshold 0.95, student is NOT flagged (0.85 < 0.95)
    res_higher_thresh = predict_risk(trained_model, df, threshold=0.95)

    assert res_lower_thresh["probability"] == res_higher_thresh["probability"]
    assert res_lower_thresh["is_at_risk"] is True
    assert res_higher_thresh["is_at_risk"] is False
