# CreditClass

A lightweight Python repository demonstrating classification techniques on real-world credit data using the UCI German Credit dataset.

![Dashboard](outputs/figures/dashboard.png)

## Overview

This project showcases six different machine learning models for credit classification tasks:

| Model | Type | Pros | Cons |
|-------|------|------|------|
| **Logistic Regression** | Linear | Interpretable, fast, good baseline | Linear decision boundary only |
| **Random Forest** | Ensemble (Bagging) | Robust, handles non-linearity | Can overfit, less interpretable |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art performance | Requires tuning |
| **SVM** | Kernel-based | Effective in high dimensions | Slow on large datasets |
| **k-NN** | Instance-based | Simple, non-parametric | Slow inference |
| **Neural Network** | Deep Learning | Learns complex patterns | Overkill for small data |

Each model includes SHAP explanations for interpretability, particularly important in regulated financial domains.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/CreditClass.git
cd CreditClass

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run the master notebook
jupyter notebook notebooks/master.ipynb
```

## Dataset

**UCI German Credit Dataset**

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Samples**: 1,000
- **Features**: 20 (7 numerical, 13 categorical)
- **Target**: Credit risk (good/bad)

The dataset describes credit applicants with features including:
- Checking account status
- Credit history
- Purpose of loan
- Credit amount
- Employment duration
- Personal information

## Project Structure

```
CreditClass/
├── src/
│   └── creditclass/
│       ├── __init__.py
│       ├── preprocessing.py      # Data loading and preprocessing
│       ├── feature_engineering.py # Feature transformations
│       ├── training.py           # Model training utilities
│       ├── evaluation.py         # Metrics and SHAP
│       └── plots.py              # Visualisation functions
├── notebooks/
│   ├── master.ipynb              # Main analysis notebook
│   ├── logistic_regression.ipynb # Logistic Regression deep-dive
│   ├── random_forest.ipynb       # Random Forest deep-dive
│   ├── xgboost.ipynb             # XGBoost deep-dive
│   ├── svm.ipynb                 # SVM deep-dive
│   ├── knn.ipynb                 # k-NN deep-dive
│   └── neural_network.ipynb      # Neural Network deep-dive
├── scripts/
│   └── generate_readme_dashboard.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_training.py
│   └── test_evaluation.py
├── data/
│   └── raw/                      # Downloaded dataset
├── outputs/
│   ├── figures/                  # Saved plots
│   └── models/                   # Serialised models
├── docs/
│   └── plans/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Usage

### Running Individual Model Notebooks

Each model has its own notebook with detailed analysis:

```bash
jupyter notebook notebooks/logistic_regression.ipynb
jupyter notebook notebooks/xgboost.ipynb
# etc.
```

### Running the Master Notebook

The master notebook provides comprehensive EDA and model comparison:

```bash
jupyter notebook notebooks/master.ipynb
```

### Generating the Dashboard

To regenerate the README dashboard figure:

```bash
python scripts/generate_readme_dashboard.py
```

### Running Tests

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=creditclass --cov-report=html
```

## Classification Tasks

The project demonstrates three classification problems derived from the same dataset:

1. **Credit Default Prediction** (binary)
   - Predict whether an applicant will default
   - Primary task for model comparison

2. **Risk Tier Classification** (multi-class)
   - Classify into low/medium/high risk tiers
   - Demonstrates multi-class metrics

3. **Loan Approval Prediction** (binary)
   - Predict approve/deny decision based on business rules
   - Shows how features serve different objectives

## Results Summary

Performance on the credit default prediction task:

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| XGBoost | ~0.75 | ~0.55 | ~0.78 |
| Random Forest | ~0.74 | ~0.52 | ~0.76 |
| Logistic Regression | ~0.73 | ~0.48 | ~0.75 |
| SVM | ~0.73 | ~0.47 | ~0.74 |
| k-NN | ~0.70 | ~0.42 | ~0.70 |
| Neural Network | ~0.72 | ~0.45 | ~0.73 |

*Note: Results may vary slightly due to random initialisation.*

## Key Findings

1. **Tree-based models** (XGBoost, Random Forest) typically perform best on this tabular dataset

2. **Important features** consistently identified:
   - Checking account status
   - Credit history
   - Credit amount
   - Duration

3. **Interpretability trade-off**: Logistic Regression offers full interpretability; tree-based models offer better performance with SHAP explanations

4. **Class imbalance**: ~30% bad credit risk affects model calibration

## Dependencies

Core:
- numpy, pandas
- scikit-learn
- xgboost
- torch
- shap
- matplotlib, seaborn

Development:
- pytest
- jupyter

See `pyproject.toml` for full list with versions.

## Licence

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
