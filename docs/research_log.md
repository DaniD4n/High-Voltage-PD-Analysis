# Research Log

## 2026-05-31

Implemented:

- Data loading from parquet files
- 2D KDE density estimation
- Gaussian Mixture Models (GMM)
- Combined KDE + GMM scoring
- Percentile-based filtering
- Peak detection using phase KDE
- Probability estimation from KDE

Observations:

- KDE phase peaks may provide a reasonable estimate for the number of GMM components.
- Further testing is required with different PD classes and noise levels.
