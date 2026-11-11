import streamlit as st
from src.data_loader import load_css, load_dataset, render_responsible_ai_notice, render_footer
from src.model_loader import is_model_available

# Page configuration
st.set_page_config(
    page_title="Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom white-based gradient styling
load_css()

# Sidebar Navigation Summary
with st.sidebar:
    st.markdown("### 🎓 **Student Intelligence**")
    st.markdown("<span style='color: #000000; font-weight:700;'>Academic Support System</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        **Application Pages:**
        - **1. Dashboard**: Performance indicators & distributions
        - **2. Student Explorer**: Filter and inspect records
        - **3. Individual Prediction**: Real-time risk scoring
        - **4. Model Performance**: Validation metrics & diagnostic curves
        """
    )
    st.markdown("---")
    
    # System Status check
    df_check = load_dataset()
    model_ready = is_model_available()
    
    st.markdown("**System Status:**")
    if df_check is not None:
        st.markdown(f"✅ Dataset: `{len(df_check):,}` records")
    else:
        st.markdown("⚠️ Dataset: *Unavailable*")
        
    if model_ready:
        st.markdown("✅ Model Artifact: `Ready`")
    else:
        st.markdown("⚠️ Model Artifact: *Not Found*")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.88rem; color: #000000;">
            Developed by <strong>Sekar Kumaran</strong><br>
            <a href="https://github.com/sekar-kumaran" target="_blank" style="color: #000000; font-weight: 600; text-decoration: underline;">
                GitHub Profile ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Page Header
st.title("🎓 Student Performance Intelligence")
st.markdown(
    "<h4 style='color: #475569; font-weight:500; margin-top:-10px;'>Predictive Academic Support & Early Risk Identification System</h4>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="glass-card">
        <h3 style="margin-top:0; color:#0F172A;">Welcome to the Academic Intelligence Portal</h3>
        <p style="color:#334155; line-height:1.6; font-size:0.96rem;">
            This system helps educators and academic counselors identify students who may need additional 
            academic support ahead of scheduled assessments. By analyzing historical exam performance, 
            attendance regularity, and assignment completion behaviors, the platform provides timely, 
            interpretable advisories before learning gaps widen.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Cards
st.markdown("### 🧭 Portal Navigation")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#FF007F; margin-top:0;">📊 1. Overview Dashboard</h4>
            <p style="color:#475569; font-size:0.92rem; line-height:1.5;">
                Explore cohort-wide metrics including attendance distributions, assignment completion patterns, 
                and overall academic indicators across 5,000 students.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#FF007F; margin-top:0;">🎯 3. Individual Prediction</h4>
            <p style="color:#475569; font-size:0.92rem; line-height:1.5;">
                Enter a student's current profile to generate an instant, calibrated risk assessment with 
                actionable support recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#FF007F; margin-top:0;">🔍 2. Student Explorer</h4>
            <p style="color:#475569; font-size:0.92rem; line-height:1.5;">
                Search, filter, and inspect individual student records by cohort, attendance range, 
                and submission punctuality.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#FF007F; margin-top:0;">📈 4. Model Performance</h4>
            <p style="color:#475569; font-size:0.92rem; line-height:1.5;">
                Review held-out evaluation results, ROC/PR curves, classification threshold tuning, 
                and permutation feature importance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Dataset & Project Specifications
st.markdown("### 📋 Project Specifications")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Student Cohort", "5,000", "Synthetic")
c2.metric("Historical Exams", "3 Assessments", "Progression")
c3.metric("At-Risk Threshold", "Score < 40", "Next Exam")
c4.metric("Horizon", "30 Days", "Forward Looking")

# Responsible AI Notice
render_responsible_ai_notice()

# Developer Attribution Footer
render_footer()
