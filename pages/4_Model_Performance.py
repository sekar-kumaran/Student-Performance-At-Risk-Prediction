import streamlit as st
from src.data_loader import load_css, load_plotly_figure, render_responsible_ai_notice, render_footer
from src.model_loader import load_metadata, is_model_available

st.set_page_config(
    page_title="Model Performance — Student Performance Intelligence",
    page_icon="📈",
    layout="wide"
)

# Apply custom styling
load_css()

# Header
st.title("Model Performance & Evaluation")
st.markdown(
    "<p style='color: #475569; font-size: 1.05rem; margin-top:-8px;'>"
    "Held-out test set evaluation metrics, diagnostic curves, and feature reliance analysis."
    "</p>",
    unsafe_allow_html=True
)

metadata = load_metadata()

if metadata is None:
    st.info(
        "ℹ️ **Model evaluation results are currently unavailable.**\n\n"
        "Please run the machine learning pipeline notebook (`notebooks/ml_pipeline.ipynb`) to train models, "
        "evaluate on the held-out test split, and generate the evaluation artifacts."
    )
    render_responsible_ai_notice()
    render_footer()
    st.stop()

# Model Architecture & Configuration Summary
model_name = metadata.get("model_name", "Random Forest")
threshold = metadata.get("classification_threshold", 0.67)
dataset_info = metadata.get("dataset_info", {})
test_metrics = metadata.get("test_metrics", {})

st.markdown("### ⚙️ Evaluation Setup")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Selected Model", model_name)
info_col2.metric("Decision Threshold", f"{threshold:.2f}")
info_col3.metric("Held-Out Test Size", f"{dataset_info.get('test_rows', 750):,} students")
info_col4.metric("Chronological Split", "70% / 15% / 15%")

# Verified Metrics Grid
st.markdown("### 🎯 Held-Out Test Evaluation Metrics")
st.markdown(
    "<p style='color: #64748B; font-size: 0.88rem; margin-top:-6px;'>"
    "Evaluated strictly once on the unseen chronological test partition (students evaluated after September 2025)."
    "</p>",
    unsafe_allow_html=True
)

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
m_col1.metric("Recall", f"{test_metrics.get('Recall', 0.0):.3f}")
m_col2.metric("Precision", f"{test_metrics.get('Precision', 0.0):.3f}")
m_col3.metric("F1-Score", f"{test_metrics.get('F1', 0.0):.3f}")
m_col4.metric("PR-AUC", f"{test_metrics.get('PR-AUC', 0.0):.3f}")
m_col5.metric("ROC-AUC", f"{test_metrics.get('ROC-AUC', 0.0):.3f}")
m_col6.metric("Accuracy", f"{test_metrics.get('Accuracy', 0.0):.3f}")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Diagnostic Charts (Tabs utilizing all saved ML figures)
st.markdown("### 📊 Diagnostic Visualizations & Model Diagnostics")

tab_cm, tab_compare, tab_curves, tab_thresh, tab_importance = st.tabs([
    "Confusion Matrix & Error Analysis",
    "Initial vs Tuned Models",
    "ROC & PR Diagnostic Curves",
    "Threshold Optimization",
    "Feature Importance"
])

with tab_cm:
    cm_col1, cm_col2 = st.columns([1.3, 1])
    
    with cm_col1:
        fig_cm = load_plotly_figure("confusion_matrix")
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix figure not found in `reports/figures/`.")

    with cm_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#0F172A; margin-top:0;">Error Interpretation</h4>
                <p style="color:#334155; font-size:0.9rem; line-height:1.6;">
                    <strong>False Positives (FP):</strong><br>
                    Students flagged for academic support who ended up scoring &ge; 40 independently. 
                    <em>Impact:</em> Low consequence (advisory counseling session).
                </p>
                <hr style="border-color: #E2E8F0; margin: 10px 0;">
                <p style="color:#334155; font-size:0.9rem; line-height:1.6;">
                    <strong style="color:#FF007F;">False Negatives (FN) — Critical:</strong><br>
                    Students who actually scored &lt; 40 but were not detected by the model.
                    <em>Impact:</em> Missed intervention window; student receives no early tutoring.
                </p>
                <hr style="border-color: #E2E8F0; margin: 10px 0;">
                <p style="color:#64748B; font-size:0.85rem; margin:0;">
                    <em>Design Goal:</em> The classification threshold was intentionally tuned on the validation set 
                    to maximize identification of genuine at-risk students.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Probability Distribution by Prediction Outcome")
    fig_errors = load_plotly_figure("error_analysis")
    if fig_errors:
        st.plotly_chart(fig_errors, use_container_width=True)
        st.caption("Figure: Predicted probability distribution comparing correctly identified at-risk students (True Positives) versus missed students (False Negatives). Red dashed line indicates the decision threshold.")
    else:
        st.info("Error analysis figure not found in `reports/figures/`.")

with tab_compare:
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.markdown("#### Initial Default Models")
        fig_init = load_plotly_figure("initial_model_comparison")
        if fig_init:
            st.plotly_chart(fig_init, use_container_width=True)
        else:
            st.info("Initial model comparison plot not found.")
            
    with cmp_col2:
        st.markdown("#### After Chronological Hyperparameter Tuning")
        fig_tuned = load_plotly_figure("tuned_model_comparison")
        if fig_tuned:
            st.plotly_chart(fig_tuned, use_container_width=True)
        else:
            st.info("Tuned model comparison plot not found.")
            
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#0F172A; margin-top:0;">Model Selection Decision Rationale</h4>
            <p style="color:#334155; font-size:0.9rem; line-height:1.6; margin:0;">
                • <strong>Baseline:</strong> The DummyClassifier achieved negligible recall (~0.05) and F1 (~0.05), confirming that non-trivial learning was required.<br>
                • <strong>Hyperparameter Optimization:</strong> RandomizedSearchCV using <code>TimeSeriesSplit(n_splits=5)</code> strictly within training data prevented temporal data leakage.<br>
                • <strong>Winning Architecture:</strong> <code>RandomForestClassifier</code> delivered the highest validation F1 and PR-AUC, combining high discriminative power with robust tree-based stability.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_curves:
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        fig_roc = load_plotly_figure("roc_curves")
        if fig_roc:
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curves plot not found.")
            
    with curve_col2:
        fig_pr = load_plotly_figure("precision_recall_curves")
        if fig_pr:
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Precision-Recall curves plot not found.")

with tab_thresh:
    fig_th = load_plotly_figure("threshold_analysis")
    if fig_th:
        st.plotly_chart(fig_th, use_container_width=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="color:#0F172A; margin-top:0;">Threshold Selection Summary</h4>
                <p style="color:#334155; font-size:0.9rem; line-height:1.6; margin:0;">
                    • Default threshold of 0.50 is suboptimal for imbalanced data (5.5% minority class).<br>
                    • Sweeping thresholds from 0.05 to 0.75 revealed an optimal F1 peak at <strong>{threshold:.2f}</strong> on the validation set.<br>
                    • This threshold guarantees actionable precision while ensuring high recall for critical student interventions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Threshold analysis plot not found.")

with tab_importance:
    imp_col1, imp_col2 = st.columns([1.6, 1])
    with imp_col1:
        fig_imp = load_plotly_figure("feature_importance")
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance plot not found.")
            
    with imp_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#0F172A; margin-top:0;">Permutation Feature Reliance</h4>
                <p style="color:#334155; font-size:0.9rem; line-height:1.6;">
                    Permutation feature importance measures the decrease in model PR-AUC when each feature column is shuffled on the validation split.
                </p>
                <hr style="border-color: #E2E8F0; margin: 10px 0;">
                <p style="color:#64748B; font-size:0.86rem; line-height:1.5;">
                    <strong style="color:#FF007F;">Reliance vs Causality:</strong><br>
                    Feature importance reflects statistical reliance of the predictive model, 
                    <strong>not direct causal proof</strong>. Assignment completion regularity and past exam averages 
                    correlate strongly with future performance without necessarily being the sole root cause.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Human Review Notice & Responsible AI
st.markdown("### 🛡️ Human-in-the-Loop Review Governance")
st.markdown(
    """
    <div class="glass-card" style="border-left: 4px solid #FF007F;">
        <h4 style="color:#0F172A; margin-top:0;">Human-Review Advisory Mandate</h4>
        <p style="color:#334155; font-size:0.92rem; line-height:1.6; margin:0;">
            All automated risk flags generated by this system must undergo human educator review prior to any 
            counseling intervention. Faculty and academic mentors are instructed to review qualitative factors 
            (such as health, personal circumstances, or course difficulty) that statistical models cannot observe.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

render_responsible_ai_notice()

# Developer Attribution Footer
render_footer()
