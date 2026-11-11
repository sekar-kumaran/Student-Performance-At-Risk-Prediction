import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_dir = str(Path(__file__).resolve().parent.parent)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

import streamlit as st
import pandas as pd
from src.data_loader import load_css, load_dataset, load_plotly_figure, render_responsible_ai_notice, render_footer

st.set_page_config(
    page_title="Student Explorer — Student Performance Intelligence",
    page_icon="🔍",
    layout="wide"
)

# Apply custom styling
load_css()

# Header
st.title("Student Explorer")
st.markdown(
    "<p style='color: #475569; font-size: 1.05rem; margin-top:-8px;'>"
    "Search, filter, and inspect verified student academic records across cohorts."
    "</p>",
    unsafe_allow_html=True
)

df = load_dataset()

if df is None:
    st.warning("⚠️ The dataset CSV is currently unavailable. Please ensure `data/generated/student_performance.csv` exists.")
    render_responsible_ai_notice()
    render_footer()
    st.stop()

# Filter Controls Section
st.markdown("### 🎛️ Filter Student Cohorts")

with st.expander("Filter and Search Controls", expanded=True):
    fc1, fc2, fc3 = st.columns([1.5, 1.2, 1.3])

    with fc1:
        search_id = st.text_input(
            "Search by Student ID",
            placeholder="e.g. STU00001",
            help="Case-insensitive search by student identification number"
        ).strip().upper()

    with fc2:
        cohort_options = ["All Cohorts"] + sorted(df["cohort_id"].dropna().unique().tolist())
        selected_cohort = st.selectbox("Cohort Period", cohort_options)

    with fc3:
        sort_options = {
            "Attendance Rate (Lowest First)": ("attendance_rate", True),
            "Attendance Rate (Highest First)": ("attendance_rate", False),
            "Assignment Completion (Lowest First)": ("assignment_completion_rate", True),
            "Assignment Completion (Highest First)": ("assignment_completion_rate", False),
            "Average Historical Score (Lowest First)": ("average_previous_score", True),
            "Student ID (Ascending)": ("student_id", True)
        }
        selected_sort = st.selectbox("Sort Order", list(sort_options.keys()))

    slider_col1, slider_col2 = st.columns(2)
    
    with slider_col1:
        min_att = float(df["attendance_rate"].min(skipna=True))
        max_att = float(df["attendance_rate"].max(skipna=True))
        att_range = st.slider(
            "Attendance Rate Range (%)",
            min_value=round(min_att, 1),
            max_value=round(max_att, 1),
            value=(round(min_att, 1), round(max_att, 1)),
            step=1.0
        )

    with slider_col2:
        min_comp = float(df["assignment_completion_rate"].min(skipna=True))
        max_comp = float(df["assignment_completion_rate"].max(skipna=True))
        comp_range = st.slider(
            "Assignment Completion Range (%)",
            min_value=round(min_comp, 1),
            max_value=round(max_comp, 1),
            value=(round(min_comp, 1), round(max_comp, 1)),
            step=1.0
        )

# Apply filters
filtered_df = df.copy()

if search_id:
    filtered_df = filtered_df[filtered_df["student_id"].str.contains(search_id, na=False)]

if selected_cohort != "All Cohorts":
    filtered_df = filtered_df[filtered_df["cohort_id"] == selected_cohort]

# Attendance filter
filtered_df = filtered_df[
    (filtered_df["attendance_rate"].isna()) |
    (filtered_df["attendance_rate"].between(att_range[0], att_range[1]))
]

# Completion filter
filtered_df = filtered_df[
    (filtered_df["assignment_completion_rate"].isna()) |
    (filtered_df["assignment_completion_rate"].between(comp_range[0], comp_range[1]))
]

# Apply sorting
sort_col, sort_asc = sort_options[selected_sort]
filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

# Filter Summary Counts
st.markdown(
    f"<p style='color:#475569; font-size:0.95rem; margin-top:8px;'>"
    f"Showing <strong>{len(filtered_df):,}</strong> of <strong>{len(df):,}</strong> students matching current criteria."
    f"</p>",
    unsafe_allow_html=True
)

# Display Table
display_columns = [
    "student_id",
    "cohort_id",
    "prediction_date",
    "attendance_rate",
    "recent_attendance_rate",
    "assignment_completion_rate",
    "missing_assignments",
    "late_submission_rate",
    "average_previous_score",
    "score_trend"
]

# Format column headers for clean display
column_labels = {
    "student_id": "Student ID",
    "cohort_id": "Cohort",
    "prediction_date": "Prediction Date",
    "attendance_rate": "Attendance (%)",
    "recent_attendance_rate": "Recent Att. (%)",
    "assignment_completion_rate": "Completion (%)",
    "missing_assignments": "Missing Tasks",
    "late_submission_rate": "Late Rate (%)",
    "average_previous_score": "Avg Past Score",
    "score_trend": "Score Trend",
}

formatted_df = filtered_df[display_columns].copy()
for col in ["attendance_rate", "recent_attendance_rate", "assignment_completion_rate", "late_submission_rate", "average_previous_score", "score_trend"]:
    if col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].round(1)

st.dataframe(
    formatted_df.rename(columns=column_labels),
    use_container_width=True,
    height=420
)

# Single Student Detailed View Drawer
st.markdown("### 📋 Student Profile Inspector")
student_list = filtered_df["student_id"].tolist()

if student_list:
    selected_stu = st.selectbox(
        "Select a student to view complete profile",
        options=student_list[:100],
        help="Select a student from the currently filtered list to inspect full academic metrics"
    )
    
    if selected_stu:
        stu_row = df[df["student_id"] == selected_stu].iloc[0]
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        att_str = f"{stu_row['attendance_rate']:.1f}%" if pd.notna(stu_row['attendance_rate']) else "N/A"
        rec_att_str = f"{stu_row['recent_attendance_rate']:.1f}%" if pd.notna(stu_row['recent_attendance_rate']) else "N/A"
        late_str = f"{stu_row['late_submission_rate']:.1f}%" if pd.notna(stu_row['late_submission_rate']) else "N/A"
        recency_str = f"{stu_row['days_since_last_submission']:.0f} days" if pd.notna(stu_row['days_since_last_submission']) else "N/A"

        with detail_col1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color:#0F172A; margin-top:0;">Identity & Timeline</h4>
                    <p style="color:#334155; font-size:0.9rem; line-height:1.7;">
                        <strong>ID:</strong> {stu_row['student_id']}<br>
                        <strong>Cohort:</strong> {stu_row['cohort_id']}<br>
                        <strong>Prediction Date:</strong> {pd.to_datetime(stu_row['prediction_date']).strftime('%Y-%m-%d')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with detail_col2:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color:#0F172A; margin-top:0;">Attendance & Punctuality</h4>
                    <p style="color:#334155; font-size:0.9rem; line-height:1.7;">
                        <strong>Overall Attendance:</strong> {att_str}<br>
                        <strong>Recent Attendance:</strong> {rec_att_str}<br>
                        <strong>Late Submissions:</strong> {stu_row['late_submissions']} ({late_str})<br>
                        <strong>Days Since Last Task:</strong> {recency_str}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with detail_col3:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color:#0F172A; margin-top:0;">Historical Assessments</h4>
                    <p style="color:#334155; font-size:0.9rem; line-height:1.7;">
                        <strong>Assessment 1:</strong> {stu_row['previous_score_1']:.1f}<br>
                        <strong>Assessment 2:</strong> {stu_row['previous_score_2']:.1f}<br>
                        <strong>Assessment 3:</strong> {stu_row['previous_score_3']:.1f}<br>
                        <strong>Average Score:</strong> {stu_row['average_previous_score']:.1f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("No students match the selected filter criteria.")

# Optional feature correlation context expander
with st.expander("📊 View Feature Correlation Reference"):
    fig_corr = load_plotly_figure("correlation_heatmap")
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Figure: Correlation matrix across academic indicators to contextualize student trends.")

render_responsible_ai_notice()

# Developer Attribution Footer
render_footer()
