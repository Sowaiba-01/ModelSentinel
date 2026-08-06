# ModelSentinel

**AI reliability & observability toolkit** — monitor, evaluate, explain, and protect machine-learning models with a single, consistent Python API.

[![PyPI](https://img.shields.io/pypi/v/modelvitals)](https://pypi.org/project/modelvitals/)
[![CI](https://github.com/Sowaiba-01/modelsentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/Sowaiba-01/modelsentinel/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Shipping a model is easy. Knowing whether it is still trustworthy in production is not. ModelSentinel answers the questions that come *after* `model.predict()`:

- Is my model still accurate, or is performance quietly degrading?
- Has the incoming data distribution drifted away from training?
- Is the input data even valid — missing values, duplicates, schema changes?
- Are my predicted probabilities calibrated, or overconfident?
- What single number tells me if this model is healthy right now?

## Install

```bash
pip install modelvitals
```

The install name is `modelvitals`; you still `import modelsentinel` in code (like `pip install scikit-learn` / `import sklearn`).

Or from source, with the dev tools:

```bash
git clone https://github.com/Sowaiba-01/modelsentinel.git
cd modelsentinel
pip install -e ".[dev]"
```

## 30-second quick start

```python
import modelsentinel as ms

monitor = ms.Monitor(task="classification", name="DeepGuard-B4")

# 1. How good are the predictions?
monitor.evaluate(y_true, y_pred, y_score)

# 2. Are the probabilities trustworthy?
monitor.calibration(y_true, y_score)

# 3. Has the data drifted since training?
monitor.detect_drift(reference_df, production_df)

# 4. Is the incoming data even clean?
monitor.profile_data(production_df)

# 5. One number for "is this model healthy?"
print(monitor.health_score())
# {'overall': 82.4, 'grade': 'GOOD', 'components': {...}}

# 6. A shareable HTML report of everything above
monitor.generate_report("model_report.html")
```

Every check is also available as a standalone function if you prefer not to use the `Monitor` facade:

```python
from modelsentinel import (
    evaluate_classification, evaluate_regression,
    calibration_report, optimal_threshold,
    profile_data, detect_drift, validate_schema, health_score,
)
```

## What's inside

| Module | Capability |
| --- | --- |
| `evaluation` | Classification & regression metrics, probability **calibration** (Brier, ECE, MCE), decision-**threshold** tuning (F1 / Youden's J) |
| `data_quality` | Missing values, duplicates, constant columns, IQR outliers, schema capture, quality score |
| `monitoring` | Data **drift** via KS test + PSI (numeric) and chi-square + Jensen-Shannon divergence (categorical), plus **schema validation** |
| `explainability` | Model-agnostic **permutation importance**, **feature-effect** curves, and framework-agnostic **Grad-CAM** |
| `adapters` | Uniform `predict` / `predict_proba` interface over any framework (`ModelAdapter`, `from_sklearn`) |
| `health` | Weighted **Model Health Score** that gracefully handles partial information |
| `reporting` | Self-contained, dependency-free **HTML report** |

## The Model Health Score

ModelSentinel rolls every check into a single, bounded score so you can alert on one number:

```
                    MODEL HEALTH
                         82 / 100   ·   GOOD

Performance       ████████░░  87
Data Quality      █████████░  93
Drift             ███████░░░  70
Reliability       ████████░░  80
```

Missing a component (say you haven't run drift yet)? The remaining weights renormalise automatically, so the score is always meaningful.

## Design goals

ModelSentinel is built to look and behave like a production open-source library, not a notebook dump: typed, documented, tested (`pytest`), linted (`ruff`), CI on every push, and zero heavyweight dependencies beyond the scientific-Python core.

## Roadmap

- **v0.1** — evaluation, drift, health score, HTML report ✅
- **v0.2** — data-quality profiling, advanced drift (PSI / KS / chi-square / JS) ✅
- **v0.3** — calibration, threshold analysis, schema validation, drift aggregation ✅
- **v0.4** — explainability (permutation importance, feature effects, Grad-CAM) + model adapters ✅
- **v0.5** — FastAPI monitoring server + real-time drift
- **v0.6+** — LLM & RAG evaluation (hallucination, faithfulness, toxicity)

> **ModelSentinel ships new versions regularly.** ⭐ Star and watch the repo to follow releases — see [CHANGELOG.md](CHANGELOG.md) for what's new in each one.

## Benchmarks

ModelSentinel is fast — the full workflow runs in well under a second on typical
tabular datasets. Reproduce with `python benchmarks/benchmark.py`; results are
written to [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Dataset | Samples × Features | Full workflow | Drift caught |
| --- | --- | ---: | --- |
| breast_cancer | 569 × 30 | ~247 ms | 5/30 shifted features flagged |
| wine | 178 × 13 | ~28 ms | 8/13 shifted features flagged |

## Case study: auditing a real deepfake detector

ModelSentinel was used to audit **DeepGuard**, an EfficientNet-B4 deepfake detector,
across two datasets built with different face-generation methods. Every number comes
straight from the model via ModelSentinel and is reproducible with the Colab notebook
in that project.

| Metric | 140k test split (held-out) | inswapper_128 set |
| --- | ---: | ---: |
| Images (real / fake) | 400 / 400 | 400 / 400 |
| Accuracy | 0.9975 | 0.9938 |
| F1 | 0.9975 | 0.9938 |
| ROC-AUC | 0.99999 | 0.9989 |
| Brier / ECE | 0.0028 / 0.0075 | 0.0086 / 0.0147 |
| **Health Score** | **99.65 (EXCELLENT)** | **99.23 (EXCELLENT)** |

Confusion matrices (rows = true `[real, fake]`): `[[398, 2], [0, 400]]` and `[[396, 4], [1, 399]]`.

The detector stays above 99% accuracy on **both** a held-out test split and a set built
with a different swap method (`inswapper_128`) — evidence it generalizes across
generation techniques, not just to its training distribution.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

MIT © Sowaiba Arshad
