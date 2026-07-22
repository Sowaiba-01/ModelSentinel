# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-27
### Added
- Probability **calibration** report (Brier score, ECE, MCE, reliability bins).
- Decision-**threshold** analysis (F1 and Youden's J sweeps).
- **Schema validation** between reference and production datasets.
- Drift **aggregation** into a dataset-level verdict and bounded score.

## [0.2.0]
### Added
- Data-quality profiling: missing values, duplicates, constant columns, IQR outliers.
- Advanced drift: PSI, KS test, chi-square, Jensen-Shannon divergence.

## [0.1.0]
### Added
- Classification & regression evaluation.
- Basic drift detection, Model Health Score, HTML report, test suite.
