# Tabular anomaly detection demo

SynthAnom is a Streamlit app that compares five detectors on numeric tabular
data: Isolation Forest, Local Outlier Factor, DBSCAN, Z-score, and IQR rules.
It combines their flags by a configurable vote and can generate additional
samples around flagged rows. **It does not process image, audio, or text
modalities**; the repository URL's historical “multimodal” name is broader than
the implemented scope.

## Run locally

Requires Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The app normally opens
at `http://localhost:8501`. No API key is required. Dependencies have minimum
versions rather than a fully pinned environment.

Choose one of three seeded synthetic examples—transaction-like, network-like, or
sensor-like data—or upload a CSV. The examples are generated locally, not real
fraud, intrusion, or equipment-failure records. Select at least one detector.
The app uses numeric columns; categorical and free-text fields are not modeled.

## Interpreting the output

- The comparison plot counts flags from each detector. Counts and agreement are
  **not accuracy measurements**. The app does not compute precision/recall against
  the synthetic ground-truth labels or evaluate a held-out dataset.
- PCA views summarize numeric features for display. A separated cluster is not
  by itself evidence of a real anomaly or failure.
- Augmentation uses the ensemble's predicted flags as labels, not verified
  anomalies. SMOTE, ADASYN, Gaussian noise, and interpolation create rows; the app
  does not train and test a downstream model to show that these rows improve it.
- The augmentation target is implemented as anomaly count divided by normal
  count, not the anomaly fraction of the final dataset.
- SMOTE and ADASYN receive a fixed random state. The Gaussian/interpolation paths
  use the global random generator, so the constructor's `random_state` alone does
  not make those paths reproducible. ADASYN can fall back to Gaussian noise.

CSV inputs need usable numeric columns. Median filling does not resolve every
edge case, such as an entirely missing column or too few usable rows. There is
no comprehensive upload-validation layer. Do not use the demo's flags as an
automated fraud, security, or maintenance decision.

## Source

- [`src/detectors.py`](src/detectors.py): scaling, detectors, voting, and PCA.
- [`src/sample_data.py`](src/sample_data.py): synthetic examples with fixed seed 42.
- [`src/synthesizer.py`](src/synthesizer.py): augmentation methods and fallbacks.
- [`src/visualizer.py`](src/visualizer.py): Plotly views.
- [`app.py`](app.py): Streamlit interface.

There is no committed test suite or saved validation run. This documentation
update was checked against source; no model-quality or browser-QA result is claimed.

## Attribution and license

Yogesh Kuchimanchi. The existing project documentation states MIT licensing;
a standalone `LICENSE` file is not currently included in this repository.
