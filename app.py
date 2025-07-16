import numpy as np
import pandas as pd
import os

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Data Preprocessing and Analysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, fbeta_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print("DDoS Attack Detection using Machine Learning")

# Load dataset
df = pd.read_csv("DDoS_dataset.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")
df = df.reset_index(drop=True)

print("Sample of Dataset")
display(df.head())

df = df.sample(frac=0.4, random_state=42)
x_values = df.select_dtypes(include=['number'])
corr_matrix = x_values.corr()

print("Correlation Heatmap")
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.show()

# Scatter plots
print("Scatter Plots")
fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
y_value = 'Packet Length'
for ax, x_value in zip(axis.flat, x_values):
    sns.scatterplot(data=df, x=x_value, y=y_value, hue='target', ax=ax)
    ax.set_title(f'{x_value.capitalize()} and {y_value.capitalize()}')
plt.tight_layout()
plt.show(fig)

# KDE plots
print("KDE Plots")
fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for ax, x_value in zip(axis.flat, x_values):
    sns.kdeplot(data=df, x=x_value, hue='target', fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show(fig)

# Histogram plots
print("Histograms")
fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for ax, x_value in zip(axis.flat, x_values):
    sns.histplot(data=df, x=x_value, hue="target", kde=True, ax=ax, bins=20, alpha=0.6)
    ax.set_title(f'Histogram of {x_value.capitalize()} by Target')
plt.tight_layout()
plt.show(fig)

df = df.drop(['Source IP','Source Port','Dest IP','Dest Port'], axis=1)

def remove_outliers(df):
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype in ['float64', 'int64']:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: x if pd.isnull(x) or (lower_limit <= x <= upper_limit) else None
            )
    return cleaned_df

df = remove_outliers(df)

def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

df = fill_missing_values(df)

le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])

numerical_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns
df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_dummies.drop('target', axis=1)
y = df_dummies['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluation
def evaluate(model_name, y_pred):
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f2_score': fbeta_score(y_test, y_pred, beta=2, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }

results = [
    evaluate('RandomForestClassifier', y_pred_rf),
    evaluate('XGBClassifier', y_pred_xgb),
    evaluate('DecisionTreeClassifier', y_pred_dt)
]

print("Model Evaluation Results")
df_model_results = pd.DataFrame(results)
display(df_model_results)

# Example Predictions
start_index = 12
num_rows = 5
example_data = X_test.iloc[start_index : start_index + num_rows]

predictions_rf_example = rf_model.predict(example_data)
predictions_xgb_example = xgb_classifier.predict(example_data)

print("Example Predictions")
print("Random Forest Predictions:", predictions_rf_example)
print("XGBoost Predictions:", predictions_xgb_example)
print("Actual Labels:", y_test.iloc[start_index : start_index + num_rows].values)

print("Interpreted Predictions (Random Forest):")
for pred in predictions_rf_example:
    print("Normal Traffic" if pred == 0 else "DDoS Attack")

print("Interpreted Predictions (XGBoost):")
for pred in predictions_xgb_example:
    print("Normal Traffic" if pred == 0 else "DDoS Attack")

# Save model
import joblib
joblib.dump(rf_model, 'ddos_model.pkl')

# Download the model file
from google.colab import files
files.download('ddos_model.pkl')


