"""
Synthetic Data Augmentation Module
Generates synthetic anomaly samples using SMOTE and statistical methods.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler


class SyntheticAugmentor:
    """Generate synthetic anomaly patterns for imbalanced datasets."""

    METHODS = ["SMOTE", "ADASYN", "Gaussian Noise", "Interpolation"]

    def __init__(self, method: str = "SMOTE", random_state: int = 42):
        self.method = method
        self.random_state = random_state

    def augment(
        self, df: pd.DataFrame, labels: np.ndarray, target_ratio: float = 0.3
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Augment minority class (anomalies) to reach target_ratio."""
        numeric = df.select_dtypes(include=[np.number]).fillna(0)
        X = numeric.values
        y = labels.copy()

        n_anomalies = int(y.sum())
        n_normal = len(y) - n_anomalies

        if n_anomalies < 2:
            return self._noise_augment(df, numeric, y, target_ratio)

        target_count = int(n_normal * target_ratio)
        if target_count <= n_anomalies:
            return df, y

        if self.method == "SMOTE" and n_anomalies >= 2:
            k = min(5, n_anomalies - 1)
            sampler = SMOTE(
                sampling_strategy={1: target_count},
                k_neighbors=k,
                random_state=self.random_state,
            )
            X_res, y_res = sampler.fit_resample(X, y)
        elif self.method == "ADASYN" and n_anomalies >= 2:
            k = min(5, n_anomalies - 1)
            try:
                sampler = ADASYN(
                    sampling_strategy={1: target_count},
                    n_neighbors=k,
                    random_state=self.random_state,
                )
                X_res, y_res = sampler.fit_resample(X, y)
            except Exception:
                return self._noise_augment(df, numeric, y, target_ratio)
        elif self.method == "Interpolation":
            return self._interpolation_augment(df, numeric, y, target_count)
        else:
            return self._noise_augment(df, numeric, y, target_ratio)

        df_res = pd.DataFrame(X_res, columns=numeric.columns)
        return df_res, y_res

    def _noise_augment(
        self, df, numeric, y, target_ratio
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Gaussian noise injection around anomaly points."""
        X = numeric.values
        anomaly_idx = np.where(y == 1)[0]
        n_normal = (y == 0).sum()
        target_count = int(n_normal * target_ratio) - len(anomaly_idx)
        if target_count <= 0:
            return df, y

        if len(anomaly_idx) == 0:
            center = X.mean(axis=0) + 3 * X.std(axis=0)
            noise = np.random.randn(target_count, X.shape[1]) * 0.5
            synthetic = center + noise
        else:
            base = X[np.random.choice(anomaly_idx, target_count, replace=True)]
            noise = np.random.randn(*base.shape) * 0.3 * X.std(axis=0)
            synthetic = base + noise

        X_aug = np.vstack([X, synthetic])
        y_aug = np.concatenate([y, np.ones(target_count)])
        df_aug = pd.DataFrame(X_aug, columns=numeric.columns)
        return df_aug, y_aug

    def _interpolation_augment(
        self, df, numeric, y, target_count
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Linear interpolation between anomaly pairs."""
        X = numeric.values
        anomaly_idx = np.where(y == 1)[0]
        n_to_gen = target_count - len(anomaly_idx)
        if n_to_gen <= 0 or len(anomaly_idx) < 2:
            return df, y

        synthetics = []
        for _ in range(n_to_gen):
            i, j = np.random.choice(anomaly_idx, 2, replace=False)
            lam = np.random.random()
            synthetics.append(lam * X[i] + (1 - lam) * X[j])

        X_aug = np.vstack([X, np.array(synthetics)])
        y_aug = np.concatenate([y, np.ones(n_to_gen)])
        df_aug = pd.DataFrame(X_aug, columns=numeric.columns)
        return df_aug, y_aug
