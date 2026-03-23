"""
Anomaly Detection Engine
Implements multiple detection algorithms with unified interface.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


class AnomalyDetector:
    """Unified anomaly detection with multiple algorithms."""

    ALGORITHMS = {
        "Isolation Forest": "_run_isolation_forest",
        "Local Outlier Factor": "_run_lof",
        "DBSCAN": "_run_dbscan",
        "Z-Score": "_run_zscore",
        "IQR": "_run_iqr",
    }

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.results = {}

    def fit_detect(self, df: pd.DataFrame, algorithms: list[str] | None = None) -> dict:
        """Run selected algorithms and return results."""
        if algorithms is None:
            algorithms = list(self.ALGORITHMS.keys())

        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("No numeric columns found in the dataset.")

        X = numeric.fillna(numeric.median())
        X_scaled = self.scaler.fit_transform(X)

        self.results = {}
        for algo in algorithms:
            if algo in self.ALGORITHMS:
                method = getattr(self, self.ALGORITHMS[algo])
                labels = method(X_scaled)
                self.results[algo] = labels

        return self.results

    def _run_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        model = IsolationForest(
            contamination=self.contamination, random_state=42, n_estimators=200
        )
        preds = model.fit_predict(X)
        return (preds == -1).astype(int)

    def _run_lof(self, X: np.ndarray) -> np.ndarray:
        model = LocalOutlierFactor(
            n_neighbors=20, contamination=self.contamination
        )
        preds = model.fit_predict(X)
        return (preds == -1).astype(int)

    def _run_dbscan(self, X: np.ndarray) -> np.ndarray:
        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X)
        return (labels == -1).astype(int)

    def _run_zscore(self, X: np.ndarray) -> np.ndarray:
        z = np.abs(stats.zscore(X, nan_policy="omit"))
        return (np.any(z > 3, axis=1)).astype(int)

    def _run_iqr(self, X: np.ndarray) -> np.ndarray:
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = np.any((X < lower) | (X > upper), axis=1)
        return mask.astype(int)

    def get_ensemble(self, threshold: float = 0.5) -> np.ndarray:
        """Majority-vote ensemble across all run algorithms."""
        if not self.results:
            raise RuntimeError("Run fit_detect first.")
        matrix = np.column_stack(list(self.results.values()))
        return (matrix.mean(axis=1) >= threshold).astype(int)

    @staticmethod
    def reduce_for_viz(X: np.ndarray, n_components: int = 3) -> np.ndarray:
        """PCA reduction for visualization."""
        pca = PCA(n_components=min(n_components, X.shape[1]))
        return pca.fit_transform(StandardScaler().fit_transform(X))
