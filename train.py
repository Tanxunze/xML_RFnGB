import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)  # For saving results and visualizations

# Load preprocessed data from fetchData.py
print("Loading preprocessed data...")
X_train = pd.read_csv('data/adult_X_train.csv')
X_test = pd.read_csv('data/adult_X_test.csv')
y_train = pd.read_csv('data/adult_y_train.csv').iloc[:, 0]
y_test = pd.read_csv('data/adult_y_test.csv').iloc[:, 0]

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Positive class ratio in training set: {y_train.mean():.4f}")
print(f"Positive class ratio in test set: {y_test.mean():.4f}")

# 1. Train Random Forest model
print("Starting Random Forest model training...")
start_time = time.time()

# Create and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=20,  # Maximum tree depth
    min_samples_split=5,  # Min samples required to split internal node
    min_samples_leaf=2,  # Min samples required at leaf node
    random_state=42,  # Random seed for reproducibility
    n_jobs=-1  # Use all available CPUs
)

# Fit the model
rf_model.fit(X_train, y_train)

# Calculate training time
rf_training_time = time.time() - start_time
print(f"Random Forest training time: {rf_training_time:.2f} seconds")

# Model evaluation
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Calculate evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"Random Forest performance metrics:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"AUC: {auc_rf:.4f}")

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest confusion matrix:")
print(cm_rf)

# Detailed classification report
print("\nRandom Forest classification report:")
print(classification_report(y_test, y_pred_rf))

# 2. Train Gradient Boosting model
print("\nStarting Gradient Boosting model training...")
start_time = time.time()

# Create and train Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Learning rate
    max_depth=5,  # Tree max depth
    min_samples_split=5,  # Min samples required to split internal node
    min_samples_leaf=2,  # Min samples required at leaf node
    random_state=42  # Random seed for reproducibility
)

# Fit the model
gb_model.fit(X_train, y_train)

# Calculate training time
gb_training_time = time.time() - start_time
print(f"Gradient Boosting training time: {gb_training_time:.2f} seconds")

# Model evaluation
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Calculate evaluation metrics
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_prob_gb)

print(f"Gradient Boosting performance metrics:")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall: {recall_gb:.4f}")
print(f"F1 Score: {f1_gb:.4f}")
print(f"AUC: {auc_gb:.4f}")

# Confusion matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("\nGradient Boosting confusion matrix:")
print(cm_gb)

# Detailed classification report
print("\nGradient Boosting classification report:")
print(classification_report(y_test, y_pred_gb))

# Save models for later use
joblib.dump(rf_model, 'results/rf_model.pkl')
joblib.dump(gb_model, 'results/gb_model.pkl')

# Performance comparison
models = ['Random Forest', 'Gradient Boosting']
accuracy = [accuracy_rf, accuracy_gb]
precision = [precision_rf, precision_gb]
recall = [recall_rf, recall_gb]
f1 = [f1_rf, f1_gb]
auc = [auc_rf, auc_gb]
training_time = [rf_training_time, gb_training_time]

# Create performance comparison table
perf_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC': auc,
    'Training Time (sec)': training_time
})

print("\nModel Performance Comparison:")
print(perf_df)

# 3. Extract and save 1000 test samples for explanation methods evaluation
# Randomly select 1000 samples
np.random.seed(42)
sample_indices = np.random.choice(X_test.shape[0], 1000, replace=False)
X_test_sample = X_test.iloc[sample_indices].reset_index(drop=True)
y_test_sample = y_test.iloc[sample_indices].reset_index(drop=True)

# Save samples for later use
X_test_sample.to_csv('data/X_test_sample.csv', index=False)
y_test_sample.to_csv('data/y_test_sample.csv', index=False)

# Visualize model performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
metrics_values = np.array([accuracy, precision, recall, f1, auc]).T

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(metrics))

plt.bar(x - bar_width / 2, metrics_values[0], bar_width, label='Random Forest')
plt.bar(x + bar_width / 2, metrics_values[1], bar_width, label='Gradient Boosting')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)
plt.savefig('results/model_performance_comparison.png')
plt.close()

# Feature importance visualization - Random Forest
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_rf)
plt.title('Random Forest - Top 15 Important Features')
plt.tight_layout()
plt.savefig('results/rf_feature_importance.png')
plt.close()

# Feature importance visualization - Gradient Boosting
feature_importance_gb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gb_model.feature_importances_
})
feature_importance_gb = feature_importance_gb.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_gb)
plt.title('Gradient Boosting - Top 15 Important Features')
plt.tight_layout()
plt.savefig('results/gb_feature_importance.png')
plt.close()

# Random Forest vs Gradient Boosting feature importance comparison
top_features = set(feature_importance_rf['feature'].head(10)).union(set(feature_importance_gb['feature'].head(10)))

# Using list-based approach instead of append
importance_data = []

for feature in top_features:
    rf_imp = feature_importance_rf[feature_importance_rf['feature'] == feature]['importance'].values
    rf_imp = rf_imp[0] if len(rf_imp) > 0 else 0

    gb_imp = feature_importance_gb[feature_importance_gb['feature'] == feature]['importance'].values
    gb_imp = gb_imp[0] if len(gb_imp) > 0 else 0

    importance_data.append({
        'feature': feature,
        'rf_importance': rf_imp,
        'gb_importance': gb_imp
    })

combined_importance = pd.DataFrame(importance_data)
combined_importance = combined_importance.sort_values('rf_importance', ascending=False)

plt.figure(figsize=(14, 10))
x = np.arange(len(combined_importance))
width = 0.35

plt.bar(x - width / 2, combined_importance['rf_importance'], width, label='Random Forest')
plt.bar(x + width / 2, combined_importance['gb_importance'], width, label='Gradient Boosting')

plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Random Forest vs Gradient Boosting - Feature Importance Comparison')
plt.xticks(x, combined_importance['feature'], rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('results/feature_importance_comparison.png')
plt.close()

print("Training and evaluation completed. Results saved to 'results' directory.")