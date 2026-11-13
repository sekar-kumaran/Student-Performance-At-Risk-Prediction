"""
Unit and integration tests for data hygiene, schema validation, and anti-leakage guardrails.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from src.data_loader import get_kpis, load_dataset
from src.prediction import prepare_student_features

# Prohibited sensitive/demographic attributes that must NEVER enter model features
PROHIBITED_SENSITIVE_ATTRIBUTES = {
    "gender",
    "sex",
    "race",
    "ethnicity",
    "caste",
    "religion",
    "socioeconomic_status",
    "ses",
    "income",
    "parent_income",
    "disability",
    "age",
    "dob",
    "date_of_birth",
    "student_name",
    "name",
    "first_name",
    "last_name",
    "email",
    "phone",
    "address",
    "nationality",
}

# Post-prediction / future / target fields that must NEVER enter model input features
PROHIBITED_FUTURE_TARGET_FIELDS = {
    "at_risk_next_assessment",
    "next_assessment_score",
    "next_assessment_date",
    "target",
    "label",
}

# Database identification keys that should not be used as learning features
PROHIBITED_IDENTIFIER_FIELDS = {
    "student_id",
    "cohort_id",
}


# ==============================================================================
# 1. Dataset Volume, Schema & Integrity Tests
# ==============================================================================

def test_dataset_loads_non_empty(raw_dataset: pd.DataFrame):
    """Verify the dataset loads cleanly as a non-empty DataFrame with >= 1,000 records."""
    assert raw_dataset is not None, "load_dataset() returned None"
    assert isinstance(raw_dataset, pd.DataFrame), "Dataset is not a pandas DataFrame"
    assert not raw_dataset.empty, "Dataset DataFrame is empty"
    assert len(raw_dataset) >= 1000, f"Expected at least 1,000 records, got {len(raw_dataset)}"
    # Specifically for our validated synthetic dataset:
    assert len(raw_dataset) == 5000, f"Expected 5,000 records, got {len(raw_dataset)}"


def test_dataset_contains_required_schema_columns(raw_dataset: pd.DataFrame):
    """Verify raw dataset contains all essential domain columns."""
    expected_columns = [
        "student_id",
        "cohort_id",
        "prediction_date",
        "attendance_rate",
        "recent_attendance_rate",
        "assignments_due",
        "assignments_completed",
        "assignment_completion_rate",
        "missing_assignments",
        "late_submissions",
        "late_submission_rate",
        "days_since_last_submission",
        "previous_score_1",
        "previous_score_2",
        "previous_score_3",
        "average_previous_score",
        "score_trend",
        "next_assessment_score",
        "at_risk_next_assessment",
    ]
    for col in expected_columns:
        assert col in raw_dataset.columns, f"Expected column '{col}' missing from dataset"


def test_dataset_target_binary_integrity(raw_dataset: pd.DataFrame):
    """Verify the target column has only binary values {0, 1} and non-trivial positive class."""
    target_values = set(raw_dataset["at_risk_next_assessment"].dropna().unique())
    assert target_values.issubset({0, 1}), f"Target contains non-binary values: {target_values}"

    # Check class imbalance is reasonable (minority at-risk class between 1% and 20%)
    at_risk_count = int(raw_dataset["at_risk_next_assessment"].sum())
    prevalence = at_risk_count / len(raw_dataset)
    assert 0.01 <= prevalence <= 0.20, f"Unexpected at-risk prevalence: {prevalence:.2%}"


def test_dataset_prediction_date_chronology(raw_dataset: pd.DataFrame):
    """Verify prediction dates are valid and span the expected operational period."""
    dates = pd.to_datetime(raw_dataset["prediction_date"])
    assert not dates.isna().any(), "Dataset contains null prediction dates"
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")


def test_get_kpis_calculation(raw_dataset: pd.DataFrame):
    """Verify KPI calculation helper computes valid, non-null indicators."""
    kpis = get_kpis(raw_dataset)
    assert isinstance(kpis, dict)
    assert kpis["total_students"] == 5000
    assert 0.0 <= kpis["avg_attendance"] <= 100.0
    assert 0.0 <= kpis["avg_completion"] <= 100.0
    assert kpis["at_risk_count"] > 0
    assert 0.0 <= kpis["at_risk_pct"] <= 100.0


def test_get_kpis_graceful_on_empty():
    """Verify get_kpis handles None and empty DataFrames safely without throwing."""
    assert get_kpis(None) == {}
    assert get_kpis(pd.DataFrame()) == {}


# 2. Ethical AI Guardrail: Zero Sensitive / Protected Attributes
def test_zero_sensitive_attributes_in_metadata_features(metadata: Dict[str, Any]):
    """
    ETHICAL AI GUARDRAIL: Assert that prohibited demographic or sensitive attributes
    are strictly absent from the model feature list.
    """
    model_features = set(metadata.get("features", []))
    for sensitive_attr in PROHIBITED_SENSITIVE_ATTRIBUTES:
        assert sensitive_attr not in model_features, (
            f"VIOLATION: Prohibited sensitive attribute '{sensitive_attr}' found in model features!"
        )


def test_zero_identifiers_in_metadata_features(metadata: Dict[str, Any]):
    """Assert database keys/identifiers are excluded to prevent memorization."""
    model_features = set(metadata.get("features", []))
    for identifier in PROHIBITED_IDENTIFIER_FIELDS:
        assert identifier not in model_features, (
            f"VIOLATION: Identifier column '{identifier}' found in model features!"
        )



# 3. Anti-Leakage Guardrail: Zero Target & Future Leakage
def test_zero_target_or_future_leakage_in_metadata_features(metadata: Dict[str, Any]):
    """
    ANTI-LEAKAGE GUARDRAIL: Assert target and future outcome columns
    are strictly absent from model training features.
    """
    model_features = set(metadata.get("features", []))
    for leak_candidate in PROHIBITED_FUTURE_TARGET_FIELDS:
        assert leak_candidate not in model_features, (
            f"LEAKAGE VIOLATION: Target or future field '{leak_candidate}' found in model features!"
        )


def test_ground_truth_target_definition_consistency(raw_dataset: pd.DataFrame):
    """
    Verify that ground-truth target is strictly defined by:
    at_risk_next_assessment = 1 if next_assessment_score < 40 else 0
    """
    expected_target = (raw_dataset["next_assessment_score"] < 40).astype(int)
    matches = (raw_dataset["at_risk_next_assessment"] == expected_target).all()
    assert matches, "Target column does not strictly match rule: next_assessment_score < 40"


# 4. Preprocessing Robustness & Outlier / Boundary Handling
def test_prepare_student_features_exact_schema(feature_order: List[str], low_risk_student_input: Dict[str, float]):
    """Verify prepare_student_features returns a DataFrame with exact feature order and 1 row."""
    df = prepare_student_features(low_risk_student_input, feature_order)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == feature_order


def test_prepare_student_features_boundary_values(feature_order: List[str], edge_case_inputs: Dict[str, Any]):
    """Verify preprocessing handles 0% and 100% boundary extremes without exceptions or NaN/Inf."""
    for boundary_name in ["all_zero", "all_hundred"]:
        raw_input = edge_case_inputs[boundary_name]
        df = prepare_student_features(raw_input, feature_order)

        assert len(df) == 1
        assert not df.isna().any().any(), f"NaN generated for boundary case '{boundary_name}'"
        assert not np.isinf(df.to_numpy()).any(), f"Inf generated for boundary case '{boundary_name}'"


def test_prepare_student_features_missing_keys_handled_gracefully(feature_order: List[str], edge_case_inputs: Dict[str, Any]):
    """Verify preprocessing gracefully fills default values when input keys are missing."""
    for missing_case in ["empty_dict", "missing_attendance"]:
        raw_input = edge_case_inputs[missing_case]
        df = prepare_student_features(raw_input, feature_order)

        assert len(df) == 1
        assert list(df.columns) == feature_order
        assert not df.isna().any().any(), f"NaN generated for missing input case '{missing_case}'"


def test_derived_feature_math_accuracy(feature_order: List[str]):
    """Verify mathematical derivations for attendance drop, average score, trend, and volatility."""
    test_input = {
        "attendance_rate": 80.0,
        "recent_attendance_rate": 60.0,
        "previous_score_1": 50.0,
        "previous_score_2": 60.0,
        "previous_score_3": 70.0,
    }
    df = prepare_student_features(test_input, feature_order)

    # attendance_drop = att - rec_att = 80 - 60 = 20.0
    assert df["attendance_drop"].iloc[0] == pytest.approx(20.0, abs=1e-3)

    # average_previous_score = mean(50, 60, 70) = 60.0
    assert df["average_previous_score"].iloc[0] == pytest.approx(60.0, abs=1e-3)

    # score_trend = s3 - s1 = 70 - 50 = 20.0
    assert df["score_trend"].iloc[0] == pytest.approx(20.0, abs=1e-3)

    # score_volatility = sample std([50, 60, 70]) = 10.0
    assert df["score_volatility"].iloc[0] == pytest.approx(10.0, abs=1e-3)

