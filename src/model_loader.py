from __future__ import annotations

import sys
import json
from pathlib import Path
import streamlit as st
import joblib

# Ensure repository root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def get_project_root() -> Path:
    """Safely locate the project root directory."""
    file_root = Path(__file__).resolve().parent.parent
    if (file_root / "artifacts").exists() or (file_root / "data").exists():
        return file_root
    current = Path.cwd()
    if (current / "artifacts").exists() or (current / "data").exists():
        return current
    elif (current.parent / "artifacts").exists() or (current.parent / "data").exists():
        return current.parent
    return file_root


@st.cache_resource(show_spinner="Loading predictive model...")
def load_model():
    """
    Load the trained model pipeline from artifacts/model.joblib.
    Returns the fitted pipeline or None if unavailable.
    """
    root = get_project_root()
    model_path = root / "artifacts" / "model.joblib"
    if not model_path.exists():
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_metadata():
    """
    Load model metadata from artifacts/metadata.json.
    Returns dictionary or None if unavailable.
    """
    root = get_project_root()
    meta_path = root / "artifacts" / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return None


def is_model_available() -> bool:
    """Check whether model artifact and metadata both exist."""
    root = get_project_root()
    has_model = (root / "artifacts" / "model.joblib").exists()
    has_meta = (root / "artifacts" / "metadata.json").exists()
    return has_model and has_meta

