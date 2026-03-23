"""
Visualization Engine
Creates stunning Plotly charts for anomaly analysis.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {"normal": "#00ff88", "anomaly": "#ff4444", "bg": "#0a0a0a", "grid": "#1a1a2e"}


def scatter_3d(X_reduced: np.ndarray, labels: np.ndarray, title: str = "3D Anomaly Space") -> go.Figure:
    """Interactive 3D scatter plot of anomalies vs normal points."""
    n_components = X_reduced.shape[1]
    color_map = np.where(labels == 1, "Anomaly", "Normal")

    if n_components >= 3:
        fig = px.scatter_3d(
            x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
            color=color_map,
            color_discrete_map={"Normal": COLORS["normal"], "Anomaly": COLORS["anomaly"]},
            opacity=0.7,
            title=title,
        )
    else:
        fig = px.scatter(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1] if n_components > 1 else np.zeros(len(X_reduced)),
            color=color_map,
            color_discrete_map={"Normal": COLORS["normal"], "Anomaly": COLORS["anomaly"]},
            opacity=0.7,
            title=title,
        )

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="white", family="JetBrains Mono, monospace"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def algorithm_comparison(results: dict, title: str = "Detection Comparison") -> go.Figure:
    """Bar chart comparing anomaly counts per algorithm."""
    algos = list(results.keys())
    counts = [int(v.sum()) for v in results.values()]

    fig = go.Figure(
        go.Bar(
            x=algos, y=counts,
            marker=dict(
                color=counts,
                colorscale=[[0, "#00ff88"], [1, "#ff4444"]],
                line=dict(width=1, color="rgba(255,255,255,0.12)"),
            ),
            text=counts, textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="white", family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], title="Anomalies Detected"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlations") -> go.Figure:
    """Correlation heatmap of numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#ff4444"], [0.5, "#0a0a0a"], [1, "#00ff88"]],
            zmin=-1, zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="white", family="JetBrains Mono, monospace"),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    return fig


def distribution_overlay(df: pd.DataFrame, labels: np.ndarray, feature: str) -> go.Figure:
    """Overlaid distributions for normal vs anomaly on a single feature."""
    normal_vals = df[feature][labels == 0]
    anomaly_vals = df[feature][labels == 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal_vals, name="Normal", marker_color=COLORS["normal"], opacity=0.6, nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=anomaly_vals, name="Anomaly", marker_color=COLORS["anomaly"], opacity=0.7, nbinsx=40,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution: {feature}",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="white", family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def ensemble_agreement(results: dict) -> go.Figure:
    """Heatmap showing agreement between algorithms on each sample."""
    matrix = np.column_stack(list(results.values()))
    sample_ids = list(range(min(200, matrix.shape[0])))
    matrix_subset = matrix[:len(sample_ids)]

    fig = go.Figure(
        go.Heatmap(
            z=matrix_subset.T,
            x=sample_ids,
            y=list(results.keys()),
            colorscale=[[0, "#1a1a2e"], [1, "#ff4444"]],
            zmin=0, zmax=1,
        )
    )
    fig.update_layout(
        title="Algorithm Agreement (first 200 samples)",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="white", family="JetBrains Mono, monospace"),
        xaxis=dict(title="Sample Index", gridcolor=COLORS["grid"]),
        margin=dict(l=120, r=20, t=50, b=40),
    )
    return fig
