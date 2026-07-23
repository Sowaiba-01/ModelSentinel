# ModelSentinel Benchmarks

- Python: 3.14.6 on Windows
- numpy 2.5.1, pandas 3.0.5, ModelSentinel 0.3.0
- Reproduce: `python benchmarks/benchmark.py`

## breast_cancer  (569 samples · 30 features)

| Check | Time (ms) | Result |
| --- | ---: | --- |
| evaluate | 7.56 | acc=0.942 f1=0.941 auc=0.992 |
| calibration | 0.18 | brier=0.036 ece=0.054 |
| threshold | 364.47 | best_t=0.28 f1=0.964 |
| data_quality | 86.48 | score=96.4 issues=2 |
| drift | 113.32 | drifted=5/30 score=83.3 |
| schema | 4.53 | valid=True |
| health_score | 0.08 | overall=92.6 (EXCELLENT) |
| **total** | **576.62** | |

Drift correctly flagged the shifted features: `mean radius, mean texture, mean perimeter, texture error, concave points error`

## wine  (178 samples · 13 features)

| Check | Time (ms) | Result |
| --- | ---: | --- |
| evaluate | 26.19 | acc=1.000 f1=1.000 auc=1.000 |
| data_quality | 39.08 | score=97.4 issues=2 |
| drift | 44.75 | drifted=8/13 score=38.5 |
| schema | 1.94 | valid=True |
| health_score | 0.04 | overall=81.2 (GOOD) |
| **total** | **112.00** | |

Drift correctly flagged the shifted features: `alcohol, malic_acid, ash, magnesium, flavanoids`
