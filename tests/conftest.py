"""
Shared pytest fixtures and configuration for Student Performance & At-Risk Prediction.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pytest

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.model_loader import load_metadata, load_model


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def metadata(project_root: Path) -> Dict[str, Any]:
    """
    Load and parse artifacts/metadata.json.
    Exposes expected feature schema, target definition, and validation/test metrics.
    """
    meta_path = project_root / "artifacts" / "metadata.json"
    assert meta_path.exists(), f"artifacts/metadata.json not found at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def feature_order(metadata: Dict[str, Any]) -> List[str]:
    """Return the exact list of 13 features expected by the trained pipeline."""
    features = metadata.get("features", [])
    assert len(features) == 13, f"Expected 13 model features, found {len(features)}"
    return features


@pytest.fixture(scope="session")
def low_risk_student_input() -> Dict[str, float]:
    """
    Valid low-risk student input profile:
    High attendance (92%), positive recent attendance (94%), high completion (95%),
    zero missing assignments, low late submission rate, and ascending assessment scores.
    """
    return {
        "attendance_rate": 92.0,
        "recent_attendance_rate": 94.0,
        "assignment_completion_rate": 95.0,
        "missing_assignments": 0,
        "late_submission_rate": 2.0,
        "days_since_last_submission": 2.0,
        "previous_score_1": 82.0,
        "previous_score_2": 86.0,
        "previous_score_3": 90.0,
    }


@pytest.fixture(scope="session")
def high_risk_student_input() -> Dict[str, float]:
    """
    Valid high-risk student input profile:
    Low attendance (42%), sharp recent attendance drop (30%), low completion (35%),
    many missing assignments (12), high late submission rate (60%), and failing scores.
    """
    return {
        "attendance_rate": 42.0,
        "recent_attendance_rate": 30.0,
        "assignment_completion_rate": 35.0,
        "missing_assignments": 12,
        "late_submission_rate": 60.0,
        "days_since_last_submission": 22.0,
        "previous_score_1": 36.0,
        "previous_score_2": 32.0,
        "previous_score_3": 28.0,
    }


@pytest.fixture(scope="session")
def borderline_student_input() -> Dict[str, float]:
    """
    Valid borderline/moderate student input profile:
    Moderate attendance (64%), mild drop (58%), 55% completion, 5 missing assignments,
    and borderline assessment scores in the 40-50s range.
    """
    return {
        "attendance_rate": 64.0,
        "recent_attendance_rate": 58.0,
        "assignment_completion_rate": 55.0,
        "missing_assignments": 5,
        "late_submission_rate": 25.0,
        "days_since_last_submission": 10.0,
        "previous_score_1": 52.0,
        "previous_score_2": 48.0,
        "previous_score_3": 44.0,
    }


@pytest.fixture(scope="session")
def edge_case_inputs() -> Dict[str, Dict[str, Any]]:
    """
    Edge-case and boundary student input profiles to test preprocessing robustness.
    """
    return {
        "all_zero": {
            "attendance_rate": 0.0,
            "recent_attendance_rate": 0.0,
            "assignment_completion_rate": 0.0,
            "missing_assignments": 0,
            "late_submission_rate": 0.0,
            "days_since_last_submission": 0.0,
            "previous_score_1": 0.0,
            "previous_score_2": 0.0,
            "previous_score_3": 0.0,
        },
        "all_hundred": {
            "attendance_rate": 100.0,
            "recent_attendance_rate": 100.0,
            "assignment_completion_rate": 100.0,
            "missing_assignments": 0,
            "late_submission_rate": 0.0,
            "days_since_last_submission": 0.0,
            "previous_score_1": 100.0,
            "previous_score_2": 100.0,
            "previous_score_3": 100.0,
        },
        "empty_dict": {},
        "missing_attendance": {
            "assignment_completion_rate": 80.0,
            "missing_assignments": 1,
            "previous_score_1": 70.0,
            "previous_score_2": 72.0,
            "previous_score_3": 75.0,
        },
        "out_of_range": {
            "attendance_rate": 150.0,
            "recent_attendance_rate": -20.0,
            "assignment_completion_rate": 200.0,
            "missing_assignments": 50,
            "late_submission_rate": 120.0,
            "days_since_last_submission": -5.0,
            "previous_score_1": 110.0,
            "previous_score_2": -10.0,
            "previous_score_3": 105.0,
        },
    }


@pytest.fixture(scope="session")
def trained_model():
    """Load and return the real trained model pipeline from artifacts/model.joblib."""
    model = load_model()
    assert model is not None, "Failed to load model from artifacts/model.joblib"
    return model


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    """Load and return the synthetic educational dataset."""
    df = load_dataset()
    assert df is not None, "Failed to load dataset from data/generated/student_performance.csv"
    return df

