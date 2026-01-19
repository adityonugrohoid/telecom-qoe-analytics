# Telecom QoE Analytics: Data Science Practice

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![POC](https://img.shields.io/badge/readiness-poc-orange.svg)

> 🔗 **Part of the Digital Twin Project** | Data Science Practice

## Production Readiness

**Level: POC**

This project demonstrates comprehensive data science methodology with:
- **Six-phase analytics pipeline** from EDA to strategic insights
- **Statistical rigor** with ANOVA and effect size analysis
- **SHAP interpretability** for model explainability
- **Production-ready models** with realistic performance metrics (R² = 0.72, ROC-AUC = 0.96)  
> Uses data from: [Telecom Digital Twin](https://github.com/adityonugrohoid/telecom-digital-twin) - Synthetic Data Generator

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Methodological Decisions](#methodological-decisions)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup & Usage](#setup--usage)
- [Author](#author)
- [License](#license)
- [Notable Code](#notable-code)

## Overview

This repository serves as a comprehensive Data Science Practice project utilizing a **synthetic telecom-digital-twin dataset**. The primary objective is to demonstrate end-to-end analytics capability—from raw data profiling and rigorous statistical testing to advanced machine learning modeling and strategic troubleshooting—focused on improving Quality of Experience (QoE) in a telecommunications network.

## Key Features

🔬 **Six-Phase Analytics Pipeline** - Structured approach from EDA to strategic insights  
📊 **Statistical Rigor** - Hypothesis testing, effect size analysis, and causal inference  
🤖 **Advanced ML Models** - XGBoost, LightGBM, clustering, and anomaly detection  
📈 **Business Translation** - Technical findings converted to actionable recommendations  
🎯 **Production-Ready Code** - Modular design with automated schema validation  

## Methodological Decisions

This project simulates a real-world "Root Cause Analysis" workflow. The modeling choices prioritize **interpretability** and **actionability** over theoretical complexity.

### 1. Model Selection: Gradient Boosting (XGBoost/LightGBM) vs. Deep Learning
* **Decision:** Utilized Tree-based ensembles (XGBoost, LightGBM) instead of Neural Networks.
* **Reasoning:** Telecom data is tabular and heterogeneous. Tree-based models natively handle non-linear feature interactions (e.g., `Congestion` × `Signal Strength`) and offer superior explainability via SHAP values. In an operational context, being able to tell a Field Engineer *why* a cell is degraded (Feature Importance) is as valuable as the prediction itself.

### 2. Interpretability: SHAP vs. Gain Metrics
* **Decision:** Adopted SHAP (SHapley Additive exPlanations) for feature attribution.
* **Reasoning:** Standard "Information Gain" metrics are biased towards high-cardinality features. SHAP provides a game-theoretic guarantees of consistency. This allowed us to prove that **Congestion** (and not just Signal Strength) was the primary driver of low QoE, directly influencing the recommendation to prioritize backhaul expansion.

### 3. Metric Selection: Recall vs. Precision
* **Decision:** Prioritized **Recall** (Sensitivity) for the Anomaly Detection model.
* **Reasoning:** In network operations, a "False Negative" (missing a major outage) is far more costly than a "False Positive" (investigating a false alarm). The model threshold was tuned to maximize the capture rate of "Low QoE" events to ensure SLA compliance.

## Prerequisites

This project requires data generated from the [Telecom Digital Twin](https://github.com/adityonugrohoid/telecom-digital-twin) repository. Generate the dataset first before running these analytics notebooks.

## Project Structure

The analysis is structured into a logical sequence of Jupyter notebooks, each addressing a specific phase of the data science lifecycle.

### 🔬 [01: Data Profiling & Exploratory QoE Landscape](notebooks/01_data_profiling_eda.ipynb)
**Goal:** Establish data trust and understand the baseline performance.
- **Schema Validation:** Automated checks ensured `users`, `cells`, and `sessions` tables were consistent for merging.
- **Missing Value Analysis:** identified network-scoped gaps vs. systematic sensor failures.
- **QoE Distribution:** Revealed the bimodal nature of user experience ('Happy' vs 'Suffering' users) and verified skewed distributions.
- **Key Insight:** Video streaming applications showed significant variability in experience compared to Chat or Web Browsing.

### 📊 [02: Statistical Analysis & Causal Inference](notebooks/02_statistical_analysis.ipynb)
**Goal:** Move beyond correlation to understand drivers of degradation.
- **Hypothesis Testing (ANOVA):** Confirmed statistically significant QoE differences between user segments (Prepaid vs. Postpaid).
- **Effect Size Analysis:** Calculated **Cohen's d** for various factors.
- **Key Insight:** Cell Congestion has a massive effect size (**d = -2.12**) on QoE, far outweighing other collected metrics. This identified congestion as the primary "villain" to fight.

### 🤖 [03: ML Regression - QoE Prediction](notebooks/03_ml_regression.ipynb)
**Goal:** Predict exact QoE scores based on network conditions.
- **Model:** XGBoost Regressor tuned with Optuna.
- **Performance:** Achieved a **Test MAE of 0.3672**, **RMSE of 0.4560**, and **R² score of 0.7247**.
- **Feature Importance:** Latency (`latency_ms`) and Congestion were identified as the most critical predictors, guiding engineering teams to focus on speed and capacity management.

### 🚦 [04: ML Classification - Degradation Prediction](notebooks/04_ml_classification.ipynb)
**Goal:** Proactively identify "Low QoE" events to trigger support or intervention.
- **Model:** LightGBM Classifier handling class imbalance.
- **Performance:** Achieved strong performance with **ROC-AUC of 0.9645**, **Precision of 0.46**, and **Recall of 0.92** for the minority "Low QoE" class.
- **Application:** This model can serve as the engine for a "Customer Experience Management" (CEM) dashboard, flagging at-risk sessions with excellent recall for proactive intervention.

### 🕵️ [05: Unsupervised Learning & Anomaly Detection](notebooks/05_unsupervised_timeseries.ipynb)
**Goal:** Detect unknown unknowns and network anomalies.
- **Technique:** STL Decomposition for time-series trend/seasonality removal, followed by Isolation Forest.
- **Findings:** Successfully isolated anomalies (~5% of data) that deviated from daily patterns.
- **Key Insight:** Anomalies frequently clustered around **5 PM (Busy Hour)**, suggesting a correlation with peak load stress testing or specific maintenance windows.

### 📑 [06: Executive Summary & Strategic Insights](notebooks/06_executive_summary.ipynb)
**Goal:** Translate technical findings into business value.
- **Strategic Recommendations:**
    1.  **Prioritize Backhaul Expansion:** Driven by the -2.12 effect size of congestion.
    2.  **Optimize Latency:** The top feature for predictive models.
    3.  **Proactive Alerts:** Deploy the Anomaly Detection model to catch evening peak failures before customers complain.
    4.  **Model Deployment:** The regression model (R² = 0.72) is suitable for real-time QoE prediction, while the classification model (ROC-AUC = 0.96) excels at identifying at-risk sessions.

## Dataset
The project uses a high-fidelity synthetic dataset generated to mimic realistic telecom network physics, encompassing:
- **Users:** Demographics, device types, and plans.
- **Cells:** Tower locations, bands (L900, L1800, L2100, etc.), and capacity.
- **Sessions:** Granular connection logs with Throughput, Latency, Jitter, Packet Loss, and calculcated QoE MOS.

## Setup & Usage (using uv)

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 1. Install uv
If you haven't already, install `uv`:
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies
Sync the environment with the `pyproject.toml` configuration:
```bash
uv sync
```

### 3. Run Notebooks
Launch the Jupyter interface within the managed environment:
```bash
uv run jupyter lab
```

## Notable Code

This repository demonstrates comprehensive data science methodology. See [NOTABLE_CODE.md](NOTABLE_CODE.md) for detailed code examples highlighting:

- Six-phase analytics pipeline implementation
- Statistical rigor with ANOVA and effect size analysis
- SHAP interpretability for model explainability

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Adityo Nugroho**  
- Portfolio: https://adityonugrohoid.github.io  
- GitHub: https://github.com/adityonugrohoid  
- LinkedIn: https://www.linkedin.com/in/adityonugrohoid/
