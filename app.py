"""
SynthAnom - Multimodal Anomaly Detection with Synthetic Augmentation
A visual dashboard for detecting anomalies across multiple algorithms
with synthetic data augmentation for class-imbalanced datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.detectors import AnomalyDetector
from src.synthesizer import SyntheticAugmentor
from src.visualizer import (
    scatter_3d, algorithm_comparison, correlation_heatmap,
    distribution_overlay, ensemble_agreement,
)
from src.sample_data import DEMO_DATASETS

# Page Config 
st.set_page_config(
    page_title="SynthAnom | Anomaly Detection",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
    .stApp { background-color: #0a0a0a; }
    section[data-testid="stSidebar"] { background-color: #0f0f0f; border-right: 1px solid #1a1a2e; }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
    .stMetric label { font-family: 'JetBrains Mono', monospace !important; color: #9ca3af !important; }
    .stMetric [data-testid="stMetricValue"] { color: #00ff88 !important; }
    div[data-testid="stMetricDelta"] { color: #00d4ff !important; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


def main():
    # Header 
    st.markdown("""
    # SynthAnom
    ### Multimodal Anomaly Detection with Synthetic Augmentation
    ---
    """)

    # Sidebar 
    with st.sidebar:
        st.markdown("## Configuration")

        data_source = st.radio("Data Source", ["Demo Dataset", "Upload CSV"])

        if data_source == "Demo Dataset":
            dataset_name = st.selectbox("Select Dataset", list(DEMO_DATASETS.keys()))
            n_samples = st.slider("Samples", 500, 5000, 2000, 100)
            anomaly_ratio = st.slider("Anomaly Ratio", 0.01, 0.20, 0.05, 0.01)
            df = DEMO_DATASETS[dataset_name](n_samples, anomaly_ratio)
            ground_truth = df.pop("is_anomaly").values if "is_anomaly" in df.columns else None
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV file to begin.")
                return
            df = pd.read_csv(uploaded)
            ground_truth = None

        st.markdown("---")
        st.markdown("### Detection Settings")
        algorithms = st.multiselect(
            "Algorithms",
            list(AnomalyDetector.ALGORITHMS.keys()),
            default=list(AnomalyDetector.ALGORITHMS.keys()),
        )
        contamination = st.slider("Contamination", 0.01, 0.20, 0.05, 0.01)
        ensemble_thresh = st.slider("Ensemble Threshold", 0.3, 0.8, 0.5, 0.05)

        st.markdown("---")
        st.markdown("### Synthetic Augmentation")
        enable_synth = st.checkbox("Enable Augmentation", value=False)
        if enable_synth:
            synth_method = st.selectbox("Method", SyntheticAugmentor.METHODS)
            target_ratio = st.slider("Target Anomaly Ratio", 0.1, 0.5, 0.3, 0.05)

    # Run Detection 
    detector = AnomalyDetector(contamination=contamination)
    results = detector.fit_detect(df, algorithms)
    ensemble = detector.get_ensemble(threshold=ensemble_thresh)

    # Synthetic Augmentation 
    if enable_synth:
        augmentor = SyntheticAugmentor(method=synth_method)
        df_aug, labels_aug = augmentor.augment(df, ensemble, target_ratio)
    else:
        df_aug, labels_aug = df, ensemble

    # Metrics Row 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Anomalies (Ensemble)", f"{int(ensemble.sum()):,}")
    with col3:
        rate = ensemble.mean() * 100
        st.metric("Anomaly Rate", f"{rate:.1f}%")
    with col4:
        st.metric("Algorithms Used", len(algorithms))

    if enable_synth:
        st.markdown("---")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Augmented Samples", f"{len(df_aug):,}", f"+{len(df_aug) - len(df):,}")
        with sc2:
            st.metric("Synthetic Anomalies", f"{int(labels_aug.sum()):,}", f"+{int(labels_aug.sum() - ensemble.sum()):,}")
        with sc3:
            st.metric("New Anomaly Ratio", f"{labels_aug.mean()*100:.1f}%")

    # Visualizations 
    st.markdown("---")
    st.markdown("## Detection Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Anomaly Space", "Algorithm Comparison", "Feature Distributions",
        "Correlation Map", "Algorithm Agreement",
    ])

    numeric = df.select_dtypes(include=[np.number]).fillna(0)

    with tab1:
        X_reduced = AnomalyDetector.reduce_for_viz(numeric.values, 3)
        fig = scatter_3d(X_reduced, ensemble)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = algorithm_comparison(results)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        feature = st.selectbox("Select Feature", numeric.columns.tolist())
        fig = distribution_overlay(df, ensemble, feature)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = ensemble_agreement(results)
        st.plotly_chart(fig, use_container_width=True)

    # Data Table 
    st.markdown("---")
    st.markdown("## Detected Anomalies")
    df_display = df.copy()
    df_display["anomaly_ensemble"] = ensemble
    for algo, labels in results.items():
        df_display[f"anomaly_{algo}"] = labels
    st.dataframe(
        df_display[df_display["anomaly_ensemble"] == 1].head(100),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
