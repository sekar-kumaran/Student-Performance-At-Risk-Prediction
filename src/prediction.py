import numpy as np
import pandas as pd


def prepare_student_features(raw_inputs: dict, feature_order: list[str]) -> pd.DataFrame:
    """
    Compute derived indicators and construct a single-row DataFrame
    matching the exact feature order expected by the trained pipeline.
    """
    att = float(raw_inputs.get("attendance_rate", 70.0))
    rec_att = float(raw_inputs.get("recent_attendance_rate", 70.0))
    att_drop = att - rec_att

    s1 = float(raw_inputs.get("previous_score_1", 60.0))
    s2 = float(raw_inputs.get("previous_score_2", 60.0))
    s3 = float(raw_inputs.get("previous_score_3", 60.0))
    scores = [s1, s2, s3]

    avg_score = float(np.mean(scores))
    score_trend = float(s3 - s1)
    score_volatility = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    comp_rate = float(raw_inputs.get("assignment_completion_rate", 70.0))
    missing = int(raw_inputs.get("missing_assignments", 2))
    late_rate = float(raw_inputs.get("late_submission_rate", 10.0))
    recency = float(raw_inputs.get("days_since_last_submission", 5.0))

    feature_map = {
        "attendance_rate": att,
        "recent_attendance_rate": rec_att,
        "attendance_drop": att_drop,
        "assignment_completion_rate": comp_rate,
        "missing_assignments": missing,
        "late_submission_rate": late_rate,
        "days_since_last_submission": recency,
        "previous_score_1": s1,
        "previous_score_2": s2,
        "previous_score_3": s3,
        "average_previous_score": avg_score,
        "score_trend": score_trend,
        "score_volatility": score_volatility,
    }

    # Build DataFrame strictly following feature_order
    row = {col: feature_map.get(col, 0.0) for col in feature_order}
    return pd.DataFrame([row])


def predict_risk(model, features_df: pd.DataFrame, threshold: float = 0.67) -> dict:
    """
    Generate real probability and binary classification advisory
    using the fitted pipeline and validated threshold.
    """
    if model is not None:
        try:
            from src.model_loader import patch_model_compatibility
            patch_model_compatibility(model)
        except Exception:
            pass

    try:
        proba = float(model.predict_proba(features_df)[:, 1][0])
    except Exception:
        # Fallback if model doesn't support predict_proba
        pred = int(model.predict(features_df)[0])
        proba = 1.0 if pred == 1 else 0.0

    is_at_risk = proba >= threshold

    # Generate tailored actionable advice
    recommendations = []
    if is_at_risk:
        recommendations.append("Priority academic tutoring recommendation before next assessment.")
        if features_df["assignment_completion_rate"].iloc[0] < 50.0:
            recommendations.append("Follow-up on missing/incomplete assignments with course mentor.")
        if features_df["attendance_drop"].iloc[0] > 10.0:
            recommendations.append("Investigate sudden attendance drop in recent weeks.")
        if features_df["score_trend"].iloc[0] < -5.0:
            recommendations.append("Address downward trajectory observed across historical exams.")
    else:
        recommendations.append("Student is currently performing on track. Maintain standard monitoring.")
        if features_df["late_submission_rate"].iloc[0] > 30.0:
            recommendations.append("Advisory: Late submissions are elevated; reinforce punctuality.")

    return {
        "probability": round(proba, 4),
        "threshold": round(threshold, 2),
        "is_at_risk": is_at_risk,
        "label": "At-Risk" if is_at_risk else "Low Risk",
        "advisory_status": "Requires Additional Academic Support" if is_at_risk else "Satisfactory Academic Progress",
        "recommendations": recommendations,
        "derived_features": {
            "attendance_drop": round(float(features_df["attendance_drop"].iloc[0]), 2),
            "average_previous_score": round(float(features_df["average_previous_score"].iloc[0]), 2),
            "score_trend": round(float(features_df["score_trend"].iloc[0]), 2),
            "score_volatility": round(float(features_df["score_volatility"].iloc[0]), 2),
        }
    }

