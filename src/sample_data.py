"""
Sample Dataset Generator
Creates demo datasets with embedded anomalies for testing.
"""

import numpy as np
import pandas as pd


def generate_credit_fraud(n_samples: int = 2000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Synthetic credit card transaction data with fraud."""
    rng = np.random.RandomState(42)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    normal = pd.DataFrame({
        "amount": rng.exponential(50, n_normal),
        "hour": rng.randint(6, 23, n_normal),
        "distance_from_home": rng.exponential(10, n_normal),
        "transaction_count_24h": rng.poisson(3, n_normal),
        "merchant_risk_score": rng.beta(2, 8, n_normal),
        "is_anomaly": 0,
    })

    anomalies = pd.DataFrame({
        "amount": rng.exponential(500, n_anomalies) + 200,
        "hour": rng.choice([0, 1, 2, 3, 4, 5], n_anomalies),
        "distance_from_home": rng.exponential(100, n_anomalies) + 50,
        "transaction_count_24h": rng.poisson(15, n_anomalies) + 5,
        "merchant_risk_score": rng.beta(8, 2, n_anomalies),
        "is_anomaly": 1,
    })

    return pd.concat([normal, anomalies]).sample(frac=1, random_state=42).reset_index(drop=True)


def generate_network_intrusion(n_samples: int = 2000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Synthetic network traffic data with intrusion attempts."""
    rng = np.random.RandomState(42)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    normal = pd.DataFrame({
        "packet_size": rng.normal(500, 100, n_normal),
        "duration": rng.exponential(2, n_normal),
        "src_bytes": rng.exponential(1000, n_normal),
        "dst_bytes": rng.exponential(800, n_normal),
        "failed_logins": rng.poisson(0.1, n_normal),
        "num_connections": rng.poisson(5, n_normal),
        "is_anomaly": 0,
    })

    anomalies = pd.DataFrame({
        "packet_size": rng.normal(1500, 300, n_anomalies),
        "duration": rng.exponential(20, n_anomalies) + 10,
        "src_bytes": rng.exponential(50000, n_anomalies),
        "dst_bytes": rng.exponential(100, n_anomalies),
        "failed_logins": rng.poisson(10, n_anomalies) + 3,
        "num_connections": rng.poisson(50, n_anomalies) + 20,
        "is_anomaly": 1,
    })

    return pd.concat([normal, anomalies]).sample(frac=1, random_state=42).reset_index(drop=True)


def generate_sensor_data(n_samples: int = 2000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Synthetic IoT sensor data with equipment failures."""
    rng = np.random.RandomState(42)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    normal = pd.DataFrame({
        "temperature": rng.normal(70, 5, n_normal),
        "vibration": rng.normal(0.5, 0.1, n_normal),
        "pressure": rng.normal(100, 10, n_normal),
        "rpm": rng.normal(3000, 200, n_normal),
        "voltage": rng.normal(220, 5, n_normal),
        "is_anomaly": 0,
    })

    anomalies = pd.DataFrame({
        "temperature": rng.normal(120, 15, n_anomalies),
        "vibration": rng.normal(2.5, 0.8, n_anomalies),
        "pressure": rng.normal(160, 20, n_anomalies),
        "rpm": rng.normal(5000, 500, n_anomalies),
        "voltage": rng.normal(260, 20, n_anomalies),
        "is_anomaly": 1,
    })

    return pd.concat([normal, anomalies]).sample(frac=1, random_state=42).reset_index(drop=True)


DEMO_DATASETS = {
    "Credit Card Fraud": generate_credit_fraud,
    "Network Intrusion": generate_network_intrusion,
    "IoT Sensor Failures": generate_sensor_data,
}
