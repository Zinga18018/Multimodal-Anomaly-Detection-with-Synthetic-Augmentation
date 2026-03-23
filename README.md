# 🔬 SynthAnom — Multimodal Anomaly Detection with Synthetic Augmentation

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-00ff88?style=flat-square" />
</p>

An interactive anomaly detection platform that benchmarks **5 detection algorithms** simultaneously and augments class-imbalanced datasets using **SMOTE, ADASYN, and statistical synthesis**. Zero API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Data Input │  │  Config Panel │  │  Viz Engine  │ │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘ │
│        │               │                 │          │
│  ┌─────▼───────────────▼─────────────────▼───────┐  │
│  │           Detection Engine                     │  │
│  │  ┌─────────┐ ┌─────┐ ┌───────┐ ┌────┐ ┌────┐ │  │
│  │  │Isolation│ │ LOF │ │DBSCAN │ │ Z  │ │IQR │ │  │
│  │  │ Forest  │ │     │ │       │ │Score│ │    │ │  │
│  │  └─────────┘ └─────┘ └───────┘ └────┘ └────┘ │  │
│  │           ▼ Ensemble (Majority Vote) ▼        │  │
│  └───────────────────────┬───────────────────────┘  │
│                          │                          │
│  ┌───────────────────────▼───────────────────────┐  │
│  │        Synthetic Augmentation Module           │  │
│  │   SMOTE │ ADASYN │ Gaussian │ Interpolation   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Features

- **5 Detection Algorithms**: Isolation Forest, Local Outlier Factor, DBSCAN, Z-Score, IQR
- **Ensemble Voting**: Majority-vote system with configurable threshold
- **Synthetic Augmentation**: SMOTE, ADASYN, Gaussian Noise, Interpolation
- **3D PCA Visualization**: Interactive scatter plots of anomaly clusters
- **Algorithm Agreement Heatmap**: See where detectors agree/disagree
- **3 Built-in Demo Datasets**: Credit Fraud, Network Intrusion, IoT Sensors
- **Upload Any CSV**: Works with arbitrary tabular data

## Quick Start

```bash
git clone https://github.com/Zinga18018/Multimodal-Anomaly-Detection-with-Synthetic-Augmentation.git
cd Multimodal-Anomaly-Detection-with-Synthetic-Augmentation
pip install -r requirements.txt
streamlit run app.py
```

## Demo Datasets

| Dataset | Features | Use Case |
|---------|----------|----------|
| **Credit Card Fraud** | amount, hour, distance, tx_count, risk_score | Financial anomalies |
| **Network Intrusion** | packet_size, duration, bytes, failed_logins | Cybersecurity |
| **IoT Sensor Failures** | temperature, vibration, pressure, rpm, voltage | Predictive maintenance |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Detection | Scikit-learn, SciPy |
| Augmentation | imbalanced-learn (SMOTE/ADASYN) |
| Visualization | Plotly |
| Dimensionality Reduction | PCA |

## License

MIT License - Yogesh Kuchimanchi
