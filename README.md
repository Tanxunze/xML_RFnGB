# Explainable AI Research Project
## Comparative Analysis of LIME and SHAP Explanation Methods

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![LIME](https://img.shields.io/badge/LIME-Latest-green.svg)](https://github.com/marcotcr/lime)
[![SHAP](https://img.shields.io/badge/SHAP-Latest-red.svg)](https://github.com/shap/shap)

## Project Overview

This research project implements and compares two leading explainable AI (XAI) methods - **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations) - for interpreting machine learning classification models. Using the UCI Adult Income dataset (48,842 samples, 97 engineered features), the project evaluates the effectiveness, performance, and interpretability characteristics of both explanation methods across Random Forest and Gradient Boosting classifiers.

## Research Objectives

- **Primary Goal**: Comparative analysis of LIME vs SHAP explanation quality and computational efficiency
- **Model Interpretation**: Generate meaningful explanations for income prediction models
- **Performance Evaluation**: Assess explanation generation time and setup overhead
- **Visualization**: Create comprehensive visual comparisons of explanation methods
- **Academic Contribution**: Provide empirical evidence for XAI method selection in classification tasks

## Key Features

- **Comprehensive Data Pipeline**: Automated data fetching, preprocessing, and feature engineering
- **Multi-Model Support**: Implementation across Random Forest and Gradient Boosting algorithms
- **Dual Explanation Methods**: Full LIME and SHAP implementation with comparative analysis
- **Performance Benchmarking**: Detailed timing analysis and computational efficiency metrics
- **Rich Visualizations**: Feature importance plots, explanation comparisons, and model performance charts
- **Reproducible Research**: Seeded random states and documented methodology for replication

## Technical Architecture

### Data Processing Pipeline
- **Dataset**: UCI Adult Income dataset (48,842 samples, 15 original features)
- **Missing Data**: Handled 2,799 workclass, 2,809 occupation, and 857 native-country missing values
- **Feature Engineering**: One-hot encoding expanded to 97 features from categorical variables
- **Train/Test Split**: Stratified 80/20 split (39,073 training, 9,769 testing samples)
- **Class Distribution**: 23.93% positive class (income >$50K), 76.07% negative class

### Machine Learning Models
- **Random Forest Classifier**: 100 estimators, max depth 20, optimized hyperparameters
- **Gradient Boosting Classifier**: 100 estimators, learning rate 0.1, max depth 5
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Explainability Implementation
- **LIME Integration**: Tabular explainer with categorical feature support
- **SHAP Integration**: TreeExplainer for efficient tree-based model interpretation
- **Sample Analysis**: Detailed explanations for representative data points
- **Global Interpretations**: Feature importance aggregation across samples

## Project Structure

```
explainable-ai-research/
├── fetchData.py              # Data acquisition and preprocessing
├── train.py                  # Model training and evaluation
├── explain.py                # XAI methods implementation
├── data/                     # Processed datasets
│   ├── adult_X_train.csv     # Training features (39,073 × 97)
│   ├── adult_X_test.csv      # Testing features (9,769 × 97)
│   ├── adult_y_train.csv     # Training labels
│   ├── adult_y_test.csv      # Testing labels
│   ├── X_test_sample.csv     # Sample for explanation analysis (1,000 samples)
│   └── y_test_sample.csv     # Corresponding labels
├── results/                  # Model outputs and analysis
│   ├── rf_model.pkl          # Trained Random Forest model
│   ├── gb_model.pkl          # Trained Gradient Boosting model
│   ├── explanations/         # Generated explanation visualizations
│   └── evaluation/           # Performance comparison results
│       └── explanation_timing.csv
└── graph/                    # Exploratory data analysis plots
    ├── feature_distributions.png
    └── feature_vs_income.png
```

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Dependencies Installation
```bash
# Core machine learning stack
pip install pandas numpy scikit-learn

# Visualization libraries
pip install matplotlib seaborn

# Explainability frameworks
pip install lime shap

# Additional utilities
pip install joblib
```

### Quick Start
```bash
# 1. Data acquisition and preprocessing
python fetchData.py

# 2. Model training and evaluation
python train.py

# 3. Explanation generation and comparison
python explain.py
```

## Methodology

### Data Preprocessing
1. **Missing Value Treatment**: Mode imputation for categorical variables
2. **Feature Encoding**: One-hot encoding for categorical features (workclass, education, etc.)
3. **Target Variable**: Binary classification (>50K vs <=50K income)
4. **Dataset Splitting**: Stratified sampling to maintain class balance

### Model Development
1. **Baseline Training**: Random Forest and Gradient Boosting with default parameters
2. **Hyperparameter Optimization**: Grid search for optimal model performance
3. **Cross-Validation**: 5-fold CV for robust performance estimation
4. **Model Persistence**: Serialization for consistent explanation analysis

### Explanation Analysis
1. **Sample Selection**: Random sampling of 1000 test instances for detailed analysis
2. **Local Explanations**: Individual prediction explanations using LIME and SHAP
3. **Global Interpretations**: Feature importance aggregation and comparison
4. **Performance Metrics**: Setup time, explanation generation time, memory usage

## Results & Findings

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | AUC | Training Time |
|-------|----------|-----------|--------|----------|-----|---------------|
| Random Forest | 0.8651 | 0.7948 | 0.5881 | 0.6760 | 0.9186 | 0.42s |
| Gradient Boosting | 0.8774 | 0.7953 | 0.6565 | 0.7193 | 0.9296 | 9.59s |

### Explanation Method Comparison
| Method | Setup Time | Avg. Explanation Time | Performance Notes |
|--------|------------|----------------------|-------------------|
| LIME-RF | 0.0160s | 0.0704s | Consistent local explanations |
| SHAP-RF | 0.0191s | 0.0640s | Theoretically grounded, stable |
| LIME-GB | 0.0160s | 0.0460s | Faster for boosting models |
| SHAP-GB | 0.0021s | 0.0004s | Extremely fast tree explanation |

### Key Insights
- **Model Comparison**: Gradient Boosting outperforms Random Forest (87.74% vs 86.51% accuracy)
- **SHAP Efficiency**: SHAP-GB provides 175x faster explanations than LIME-GB (0.0004s vs 0.046s)
- **Feature Engineering Impact**: One-hot encoding expanded dataset from 15 to 97 features
- **Class Imbalance**: 76% negative class requires careful evaluation of precision-recall trade-offs
- **Explanation Consistency**: Both LIME and SHAP identify similar feature importance patterns
- **Training Efficiency**: Random Forest trains 23x faster than Gradient Boosting (0.42s vs 9.59s)

## Generated Visualizations

The project produces comprehensive visual outputs:
- **Feature Importance Rankings**: Comparative analysis across models and methods
- **Individual Explanations**: Sample-specific prediction explanations
- **Global Summary Plots**: Aggregated feature impact analysis
- **Performance Comparisons**: Model accuracy and explanation timing benchmarks
- **Distribution Analysis**: Dataset characteristics and class balance visualization

## Academic Applications

This project demonstrates proficiency in:
- **Machine Learning Engineering**: End-to-end ML pipeline development
- **Research Methodology**: Systematic comparison of explanation methods
- **Data Science**: Large-scale dataset processing and analysis
- **Visualization**: Professional-quality scientific plotting
- **Reproducible Research**: Version-controlled, documented analysis workflow

## Future Enhancements

- **Deep Learning Integration**: Extend to neural network architectures
- **Additional XAI Methods**: Integrate Anchors, Counterfactual explanations
- **Interactive Dashboard**: Web-based explanation exploration interface
- **Fairness Analysis**: Bias detection and mitigation in explanations
- **Real-time Inference**: Production-ready explanation service deployment

## Contributing

Research contributions are welcome:
1. Fork the repository
2. Create feature branch for new analysis
3. Follow PEP 8 coding standards
4. Add comprehensive documentation
5. Submit pull request with detailed methodology

## License

This research project is licensed under the MIT License - see LICENSE file for details.

## Contact

For research inquiries or collaboration opportunities:
- **Research Focus**: Explainable AI, Machine Learning Interpretability
- **Technical Stack**: Python, Scikit-learn, LIME, SHAP
- **Applications**: Graduate program applications, Research internships
