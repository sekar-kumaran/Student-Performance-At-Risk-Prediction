from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.io as pio


def get_project_root() -> Path:
    """Safely locate the project root directory."""
    current = Path.cwd()
    if (current / "data").exists() or (current / "reports").exists():
        return current
    elif (current.parent / "data").exists() or (current.parent / "reports").exists():
        return current.parent
    return current


def load_css():
    """Inject custom CSS stylesheet."""
    root = get_project_root()
    css_path = root / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading student records...")
def load_dataset() -> pd.DataFrame | None:
    """
    Load the synthetic educational dataset.
    Returns DataFrame if found, or None with a friendly warning.
    """
    root = get_project_root()
    data_path = root / "data" / "generated" / "student_performance.csv"
    if not data_path.exists():
        return None
    try:
        df = pd.read_csv(data_path, parse_dates=["prediction_date"])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_plotly_figure(filename: str):
    """
    Load a pre-saved Plotly figure from reports/figures/ by filename.
    Appends .json if not present.
    Returns Plotly Figure or None if unavailable.
    """
    root = get_project_root()
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    fig_path = root / "reports" / "figures" / filename
    if not fig_path.exists():
        return None
    try:
        # Load raw bytes first to handle encodings robustly
        raw_bytes = open(fig_path, "rb").read()
        try:
            json_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            json_str = raw_bytes.decode("cp1252", errors="replace")

        fig = pio.from_json(json_str)

        # Style layout for the clean white-based light theme
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#1E293B", family="Segoe UI, Roboto, Helvetica, Arial, sans-serif"),
            title=dict(font=dict(color="#0F172A", size=15, family="Segoe UI, Roboto, Helvetica, Arial, sans-serif")),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        # Update axis styling for light background readability
        fig.update_xaxes(
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#0F172A")
        )
        fig.update_yaxes(
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#0F172A")
        )
        return fig
    except Exception as e:
        st.warning(f"Note: Could not render figure {filename}: {e}")
        return None


def get_kpis(df: pd.DataFrame) -> dict:
    """Calculate key academic indicators from the dataset."""
    if df is None or df.empty:
        return {}
    
    total = len(df)
    avg_att = df["attendance_rate"].dropna().mean()
    avg_rec_att = df["recent_attendance_rate"].dropna().mean()
    avg_comp = df["assignment_completion_rate"].dropna().mean()
    avg_missing = df["missing_assignments"].dropna().mean()
    
    at_risk_count = int(df["at_risk_next_assessment"].sum()) if "at_risk_next_assessment" in df.columns else 0
    at_risk_pct = (at_risk_count / total * 100) if total > 0 else 0.0

    return {
        "total_students": total,
        "avg_attendance": round(avg_att, 1),
        "avg_recent_attendance": round(avg_rec_att, 1),
        "avg_completion": round(avg_comp, 1),
        "avg_missing": round(avg_missing, 1),
        "at_risk_count": at_risk_count,
        "at_risk_pct": round(at_risk_pct, 1),
    }


def render_responsible_ai_notice():
    """Render the standard Responsible AI notice banner required by the project."""
    st.markdown(
        """
        <div class="responsible-ai-banner">
            <strong>Responsible AI & Ethical Use Notice:</strong><br>
            Model outputs are intended to support academic review. They are not definitive judgments about students. 
            Teachers and academic reviewers must consider additional context before taking action. 
            The model must not be used for automated punishment, grading or admission decisions.
            <br><em>Note: This application operates on a synthetic educational dataset (5,000 records).</em>
        </div>
        """,
        unsafe_allow_html=True
    )
