import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# fetch data from url
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# read data
adult_train = pd.read_csv(url_train, header=None, names=column_names,
                          sep=', ', engine='python')
adult_test = pd.read_csv(url_test, header=None, names=column_names,
                         sep=', ', engine='python', skiprows=1)

# remove point of the end
adult_test['income'] = adult_test['income'].str.rstrip('.')

# merge data
adult_data = pd.concat([adult_train, adult_test])

print(f"Dataset size: {adult_data.shape}")

print("Basic info of dataset:")
print(adult_data.info())

# category distribution
print("\nIncome distribution:")
print(adult_data['income'].value_counts(normalize=True))

# Basic statistics for numerical characteristics
print("\nNumerical characteristics statistics:")
print(adult_data.describe())

# Check missing values
print("\nNumber of missing values per feature:")
print(adult_data.isnull().sum())

# check the column containg '?'
for column in adult_data.columns:
    count = (adult_data[column] == '?').sum()
    if count > 0:
        print(f"The feature '{column}' contains {count} '?' values")

# process missing values
categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

# replace '?' with 'np.nan'
for feature in categorical_features:
    adult_data[feature] = adult_data[feature].replace('?', np.nan)

# fill missing values from most frequent values
for feature in categorical_features:
    most_frequent = adult_data[feature].mode()[0]
    adult_data[feature] = adult_data[feature].fillna(most_frequent)

# Convert target variable to binary
adult_data['income'] = adult_data['income'].map({'>50K': 1, '<=50K': 0})

# process categorical features - use of unique heat coding
# Creating dummy variables （the value only contain 0 or 1)
adult_encoded = pd.get_dummies(adult_data, columns=categorical_features, drop_first=True)

print(f"Size of the encoded dataset: {adult_encoded.shape}")
print("Encoded features:")
print(adult_encoded.columns.tolist())

# Randomly divide the dataset into a training set (80%) and a test set (20%)
X = adult_encoded.drop('income', axis=1)
y = adult_encoded['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")
print(f"Proportion of positive examples in the training set: {y_train.mean():.4f}")
print(f"Proportion of positive examples in the test set: {y_test.mean():.4f}")

# save processed data
X_train.to_csv('./data/adult_X_train.csv', index=False)
X_test.to_csv('./data/adult_X_test.csv', index=False)
y_train.to_csv('./data/adult_y_train.csv', index=False)
y_test.to_csv('./data/adult_y_test.csv', index=False)

# distribution of numerical features
numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                   'capital-loss', 'hours-per-week']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(adult_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('./graph/feature_distributions.png')
plt.close()

# See how certain characteristics relate to the target variable
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['age', 'education-num', 'hours-per-week']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='income', y=feature, data=adult_data)
    plt.title(f'{feature} vs Income')
plt.tight_layout()
plt.savefig('./graph/feature_vs_income.png')
plt.close()