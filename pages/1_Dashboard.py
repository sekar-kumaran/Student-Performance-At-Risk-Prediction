import streamlit as st
from src.data_loader import (
    load_css,
    load_dataset,
    load_plotly_figure,
    get_kpis,
    render_responsible_ai_notice
)

st.set_page_config(
    page_title="Dashboard — Student Performance Intelligence",
    page_icon="📊",
    layout="wide"
)

# Apply custom styling
load_css()

# Header
st.title("Student Performance Overview")
st.markdown(
    "<p style='color: #475569; font-size: 1.05rem; margin-top:-8px;'>"
    "This dashboard provides an overview of student attendance, assignment completion and academic engagement signals."
    "</p>",
    unsafe_allow_html=True
)

# Load dataset
df = load_dataset()

if df is None:
    st.warning("⚠️ The dataset CSV (`data/generated/student_performance.csv`) is currently unavailable. Please generate or place the dataset file in the project directory.")
    render_responsible_ai_notice()
    st.stop()

# KPI Metrics Section
kpis = get_kpis(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Students", f"{kpis['total_students']:,}")
col2.metric("Avg Attendance", f"{kpis['avg_attendance']}%")
col3.metric("Recent Attendance", f"{kpis['avg_recent_attendance']}%")
col4.metric("Avg Assignment Comp.", f"{kpis['avg_completion']}%")
col5.metric("Avg Missing Tasks", f"{kpis['avg_missing']}")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Visualizations Section
st.markdown("### 📈 Academic Indicators & Dataset Visualizations")
st.markdown(
    "<p style='color: #64748B; font-size: 0.9rem;'>"
    "Interactive figures loaded directly from validated Plotly JSON artifacts in <code>reports/figures/</code>."
    "</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "Academic Distributions",
    "Engagement & Risk Status",
    "Feature Correlation Heatmap"
])

with tab1:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_att = load_plotly_figure("attendance_distribution")
        if fig_att:
            st.plotly_chart(fig_att, use_container_width=True)
        else:
            st.info("Attendance distribution plot is currently unavailable.")

    with chart_col2:
        fig_comp = load_plotly_figure("assignment_completion_distribution")
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Assignment completion distribution plot is currently unavailable.")

with tab2:
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig_scatter = load_plotly_figure("attendance_vs_assignment")
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Attendance vs Completion scatter plot is currently unavailable.")

    with chart_col4:
        fig_dist = load_plotly_figure("class_distribution")
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Class distribution plot is currently unavailable.")

with tab3:
    fig_corr = load_plotly_figure("correlation_heatmap")
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(
            """
            <div class="glass-card" style="margin-top:10px;">
                <h4 style="color:#0F172A; margin-top:0;">Understanding Correlation Patterns</h4>
                <p style="color:#475569; font-size:0.9rem; line-height:1.6; margin:0;">
                    • <strong>Attendance & Assignment Completion:</strong> Exhibit strong positive correlation with historical assessment scores.<br>
                    • <strong>Missing Assignments & Late Submissions:</strong> Inversely correlated with next assessment score, serving as critical leading indicators for academic support.<br>
                    • <strong>Score Trend & Average Historical Score:</strong> High consistency across historical scores provides reliable baseline stability.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Correlation heatmap is currently unavailable in `reports/figures/`.")

# Responsible AI Notice Banner
render_responsible_ai_notice()
