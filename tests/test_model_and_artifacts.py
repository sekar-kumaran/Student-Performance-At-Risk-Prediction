"""
Unit and integration tests for model artifacts, metadata integrity, and evaluation metric defensibility.
"""

from __future__ import annotations
from typing import Any, Dict, List
import pytest
from sklearn.pipeline import Pipeline
from src.model_loader import is_model_available, load_metadata, load_model

# 1. Metadata Completeness & Schema Validation
def test_metadata_file_completeness(metadata: Dict[str, Any]):
    """Verify artifacts/metadata.json contains all required architectural and evaluation keys."""
    required_keys = [
        "model_name",
        "features",
        "features_with_missing",
        "target",
        "target_definition",
        "classification_threshold",
        "random_seed",
        "training_date",
        "dataset_info",
        "best_cv_params",
        "best_cv_pr_auc",
        "validation_metrics",
        "test_metrics",
    ]
    for key in required_keys:
        assert key in metadata, f"Required metadata key '{key}' is missing from metadata.json"


def test_metadata_feature_count_and_types(metadata: Dict[str, Any]):
    """Verify metadata features list contains exactly 13 distinct string features."""
    features = metadata["features"]
    assert isinstance(features, list)
    assert len(features) == 13, f"Expected 13 features, got {len(features)}"
    assert len(set(features)) == 13, "Duplicate feature names detected in metadata.json"
    assert all(isinstance(f, str) for f in features)


def test_metadata_dataset_split_accounting(metadata: Dict[str, Any]):
    """Verify chronological split rows in metadata sum up exactly to the total dataset size."""
    info = metadata["dataset_info"]
    total = info["total_rows"]
    train = info["train_rows"]
    val = info["val_rows"]
    test = info["test_rows"]

    assert total == 5000
    assert train == 3500  # 70%
    assert val == 750     # 15%
    assert test == 750    # 15%
    assert train + val + test == total, "Split partition row counts do not sum to total_rows"


# 2. Real Model Artifact Loading & Interface Compatibility
def test_is_model_available_flag():
    """Verify is_model_available() returns True when artifacts exist on disk."""
    assert is_model_available() is True


def test_load_metadata_helper_returns_dict():
    """Verify load_metadata() returns a non-empty dictionary matching metadata.json."""
    meta = load_metadata()
    assert isinstance(meta, dict)
    assert meta.get("model_name") == "RF_Tuned"


def test_real_model_artifact_loading(trained_model):
    """
    Verify src/model_loader.py loads a real, fitted Scikit-learn Pipeline
    from disk (not a mock or placeholder).
    """
    assert trained_model is not None, "load_model() returned None"
    assert isinstance(trained_model, Pipeline), f"Expected Pipeline, got {type(trained_model)}"
    assert hasattr(trained_model, "steps"), "Trained model has no steps attribute"
    assert len(trained_model.steps) >= 2, "Pipeline should contain at least preprocessor and classifier"


def test_model_exposes_predict_and_predict_proba(trained_model):
    """Verify the loaded model object exposes callable predict() and predict_proba() methods."""
    assert hasattr(trained_model, "predict"), "Model artifact missing predict() method"
    assert callable(trained_model.predict), "predict is not callable"
    assert hasattr(trained_model, "predict_proba"), "Model artifact missing predict_proba() method"
    assert callable(trained_model.predict_proba), "predict_proba is not callable"



# 3. Metric Defensibility & Range Sanity Checks


def test_evaluation_metrics_defensibility(metadata: Dict[str, Any]):
    """
    Assert that validation and held-out test evaluation metrics
    are valid floats strictly within realistic, defensible bounds [0.0, 1.0].
    """
    for split_name in ["validation_metrics", "test_metrics"]:
        split_metrics = metadata.get(split_name, {})
        assert split_metrics, f"Missing metrics dictionary for {split_name}"

        for metric_name in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
            val = split_metrics.get(metric_name)
            assert isinstance(val, (int, float)), f"{metric_name} in {split_name} is not numeric: {val}"
            assert 0.0 <= val <= 1.0, f"{metric_name} in {split_name} ({val}) is outside [0.0, 1.0]"

            # Sanity checks: precision, recall, and F1 on real imbalanced data should be non-trivial
            assert val > 0.0, f"{metric_name} in {split_name} should not be zero in a tuned model"


def test_classification_threshold_bounds(metadata: Dict[str, Any]):
    """Verify that tuned decision threshold is bounded within (0.0, 1.0)."""
    threshold = metadata.get("classification_threshold")
    assert isinstance(threshold, (int, float))
    assert 0.0 < threshold < 1.0, f"Threshold {threshold} is outside (0.0, 1.0)"
    assert threshold == pytest.approx(0.67, abs=1e-2)


def test_test_set_beats_baseline(metadata: Dict[str, Any]):
    """
    Defensibility check: The final tuned model on held-out test data
    must substantially outperform a naive random baseline (ROC-AUC > 0.5, PR-AUC > 0.1).
    """
    test_metrics = metadata["test_metrics"]
    assert test_metrics["ROC-AUC"] > 0.85, f"ROC-AUC too low: {test_metrics['ROC-AUC']}"
    assert test_metrics["PR-AUC"] > 0.40, f"PR-AUC too low: {test_metrics['PR-AUC']}"
    assert test_metrics["F1"] > 0.35, f"F1 score too low: {test_metrics['F1']}"

