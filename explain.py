import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

os.makedirs('results/explanations', exist_ok=True)
os.makedirs('results/evaluation', exist_ok=True)

# load models
print("Loading models and test samples...")
rf_model = joblib.load('results/rf_model.pkl')
gb_model = joblib.load('results/gb_model.pkl')

X_test_sample = pd.read_csv('data/X_test_sample.csv')
y_test_sample = pd.read_csv('data/y_test_sample.csv').iloc[:, 0]

# Simple checking - verifying that the model and data are loaded correctly
y_pred_rf = rf_model.predict(X_test_sample)
y_pred_gb = gb_model.predict(X_test_sample)

print(f"Random Forest accuracy on sample: {accuracy_score(y_test_sample, y_pred_rf):.4f}")
print(f"Gradient Boosting accuracy on sample: {accuracy_score(y_test_sample, y_pred_gb):.4f}")

# Get dataset feature name and data type information
feature_names = X_test_sample.columns.tolist()
categorical_features = [i for i, col in enumerate(feature_names) if 'workclass_' in col or
                        'education_' in col or 'marital-status_' in col or 'occupation_' in col or
                        'relationship_' in col or 'race_' in col or 'sex_' in col or
                        'native-country_' in col]

print(f"Total features: {len(feature_names)}")
print(f"Categorical features: {len(categorical_features)}")

# 1. LIME Implementation
print("\n*** Implementing LIME Explanations ***")
from lime import lime_tabular

# Create the LIME Interpreter
start_time = time.time()
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_test_sample.values,
    feature_names=feature_names,
    class_names=['<=50K', '>50K'],
    categorical_features=categorical_features,
    mode='classification',
    random_state=42
)
lime_setup_time = time.time() - start_time
print(f"LIME explainer setup time: {lime_setup_time:.2f} seconds")

# Selection of a small number of samples for detailed visualisation
sample_indices = np.random.choice(len(X_test_sample), 5, replace=False)

# LIME explanation times
lime_rf_times = []
lime_gb_times = []

# Example of creating a LIME interpretation for a random forest model
print("Generating LIME explanations for Random Forest model...")
for i, idx in enumerate(sample_indices):
    instance = X_test_sample.iloc[idx].values
    start_time = time.time()
    exp = lime_explainer.explain_instance(
        instance,
        rf_model.predict_proba,
        num_features=10
    )
    lime_rf_times.append(time.time() - start_time)

    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f'results/explanations/lime_rf_sample_{i}.png')
    plt.close()

# Example of creating a LIME interpretation for a gradient lifter model
print("Generating LIME explanations for Gradient Boosting model...")
for i, idx in enumerate(sample_indices):
    instance = X_test_sample.iloc[idx].values
    start_time = time.time()
    exp = lime_explainer.explain_instance(
        instance,
        gb_model.predict_proba,
        num_features=10
    )
    lime_gb_times.append(time.time() - start_time)

    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f'results/explanations/lime_gb_sample_{i}.png')
    plt.close()

print(f"Average LIME explanation time for RF: {np.mean(lime_rf_times):.4f} seconds")
print(f"Average LIME explanation time for GB: {np.mean(lime_gb_times):.4f} seconds")

# 2. SHAP Implementation
print("\n*** Implementing SHAP Explanations ***")
import shap

# Creating a SHAP interpreter for random forests model
print("Setting up SHAP explainer for Random Forest...")
start_time = time.time()
# Using the Tree interpreter for tree models (faster)
shap_rf_explainer = shap.TreeExplainer(rf_model)
shap_rf_setup_time = time.time() - start_time
print(f"SHAP RF explainer setup time: {shap_rf_setup_time:.2f} seconds")

# Creating a SHAP interpreter for gradient lifters
print("Setting up SHAP explainer for Gradient Boosting...")
start_time = time.time()
shap_gb_explainer = shap.TreeExplainer(gb_model)
shap_gb_setup_time = time.time() - start_time
print(f"SHAP GB explainer setup time: {shap_gb_setup_time:.2f} seconds")

# Generating SHAP Explanations for Random Forests
print("Calculating SHAP values for Random Forest (sample)...")
start_time = time.time()
shap_values_rf = shap_rf_explainer.shap_values(X_test_sample.iloc[sample_indices])
shap_rf_time = time.time() - start_time
print(f"SHAP RF explanation time: {shap_rf_time:.2f} seconds")

# Generating SHAP explanations for gradient lifters
print("Calculating SHAP values for Gradient Boosting (sample)...")
start_time = time.time()
shap_values_gb = shap_gb_explainer.shap_values(X_test_sample.iloc[sample_indices])
shap_gb_time = time.time() - start_time
print(f"SHAP GB explanation time: {shap_gb_time:.2f} seconds")

# Generating SHAP visualisations for random forests - using summary_plot
print("Generating SHAP visualizations...")
plt.figure(figsize=(12, 8))

if isinstance(shap_values_rf, list):
    # For dichotomisation,usually chose the SHAP value for the second category (index 1)
    shap.summary_plot(shap_values_rf[1], X_test_sample.iloc[sample_indices],
                      feature_names=feature_names, plot_type="dot", show=False)
else:
    shap.summary_plot(shap_values_rf, X_test_sample.iloc[sample_indices],
                      feature_names=feature_names, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('results/explanations/shap_rf_sample_summary.png')
plt.close()

# Generating SHAP visualisations for gradient lifters
plt.figure(figsize=(12, 8))
if isinstance(shap_values_gb, list):
    shap.summary_plot(shap_values_gb[1], X_test_sample.iloc[sample_indices],
                      feature_names=feature_names, plot_type="dot", show=False)
else:
    shap.summary_plot(shap_values_gb, X_test_sample.iloc[sample_indices],
                      feature_names=feature_names, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('results/explanations/shap_gb_sample_summary.png')
plt.close()

# Generating Summary Plots (Global Interpretation) - Random Forests
plt.figure(figsize=(10, 8))

# Calculate global summaries using more samples
shap_values_summary_rf = shap_rf_explainer.shap_values(X_test_sample.iloc[:100])

if isinstance(shap_values_summary_rf, list):
    shap.summary_plot(shap_values_summary_rf[1], X_test_sample.iloc[:100],
                      feature_names=feature_names, plot_type="bar", show=False)
else:
    shap.summary_plot(shap_values_summary_rf, X_test_sample.iloc[:100],
                      feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('results/explanations/shap_rf_global_summary.png')
plt.close()

# Generating Summary Graphs (Global Interpretation) - Gradient Booster
plt.figure(figsize=(10, 8))
shap_values_summary_gb = shap_gb_explainer.shap_values(X_test_sample.iloc[:100])

if isinstance(shap_values_summary_gb, list):
    shap.summary_plot(shap_values_summary_gb[1], X_test_sample.iloc[:100],
                      feature_names=feature_names, plot_type="bar", show=False)
else:
    shap.summary_plot(shap_values_summary_gb, X_test_sample.iloc[:100],
                      feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('results/explanations/shap_gb_global_summary.png')
plt.close()


# Generate separate SHAP interpretation plots for each sample
for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(10, 6))

    # debug
    print(f"SHAP values type: {type(shap_values_rf)}")
    if isinstance(shap_values_rf, list):
        print(f"SHAP values RF[1] shape: {np.array(shap_values_rf[1]).shape}")
    else:
        print(f"SHAP values RF shape: {np.array(shap_values_rf).shape}")

    # Extracting and visualising SHAP values
    if isinstance(shap_values_rf, list):
        sample_shap_values = shap_values_rf[1][i]
        if len(sample_shap_values.shape) > 1:
            sample_shap_values = sample_shap_values[:, 0]  # use first column

        # fetch important feature
        feature_importance = np.abs(sample_shap_values)
        top_n = min(10, len(feature_importance))  # preventinsufficient number of features
        sorted_idx = np.argsort(feature_importance)[-top_n:]

        # Create Bar Charts
        plt.barh(np.arange(len(sorted_idx)), sample_shap_values[sorted_idx])
        plt.yticks(np.arange(len(sorted_idx)), [feature_names[j][:20] for j in sorted_idx])
        plt.title(f'SHAP values for RF - Sample {i}')
    else:
        sample_shap_values = shap_values_rf[i]
        if len(sample_shap_values.shape) > 1:
            sample_shap_values = sample_shap_values[:, 0]

        feature_importance = np.abs(sample_shap_values)
        top_n = min(10, len(feature_importance))
        sorted_idx = np.argsort(feature_importance)[-top_n:]

        plt.barh(np.arange(len(sorted_idx)), sample_shap_values[sorted_idx])
        plt.yticks(np.arange(len(sorted_idx)), [feature_names[j][:20] for j in sorted_idx])
        plt.title(f'SHAP values for RF - Sample {i}')

    plt.tight_layout()
    plt.savefig(f'results/explanations/shap_rf_sample_{i}_bar.png')
    plt.close()

# Create a separate sample chart for gradient lifters
for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values_gb, list):
        feature_names_short = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]
        feature_importance = np.abs(shap_values_gb[1][i])
        sorted_idx = np.argsort(feature_importance)[-10:]

        plt.barh(range(len(sorted_idx)), shap_values_gb[1][i][sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names_short[j] for j in sorted_idx])
        plt.title(f'SHAP values for GB - Sample {i}')
    else:
        feature_names_short = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]
        feature_importance = np.abs(shap_values_gb[i])
        sorted_idx = np.argsort(feature_importance)[-10:]

        plt.barh(range(len(sorted_idx)), shap_values_gb[i][sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names_short[j] for j in sorted_idx])
        plt.title(f'SHAP values for GB - Sample {i}')

    plt.tight_layout()
    plt.savefig(f'results/explanations/shap_gb_sample_{i}_bar.png')
    plt.close()

# 3. Assessing the multiple dimensions of interpretive methods
print("\n*** Evaluating Explanation Methods ***")

# Create a results table
explanation_results = pd.DataFrame({
    'Method': ['LIME-RF', 'LIME-GB', 'SHAP-RF', 'SHAP-GB'],
    'Setup Time (sec)': [lime_setup_time, lime_setup_time, shap_rf_setup_time, shap_gb_setup_time],
    'Explanation Time (sec)': [np.mean(lime_rf_times), np.mean(lime_gb_times),
                               shap_rf_time / len(sample_indices), shap_gb_time / len(sample_indices)]
})

# save assessment result
explanation_results.to_csv('results/evaluation/explanation_timing.csv', index=False)
print(explanation_results)

print("\nExplanation methods implementation complete.")
