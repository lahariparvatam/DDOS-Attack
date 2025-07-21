import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, fbeta_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

st.title("DDoS Attack Detection using Machine Learning")

# Load dataset
try:
    df = pd.read_csv("DDoS_dataset.csv")
    st.success(f"Dataset loaded successfully. Shape: {df.shape}")
    st.subheader("Sample of Dataset")
    st.write(df.head())
except FileNotFoundError:
    st.error("DDoS_dataset.csv not found. Please upload the dataset.")
    st.stop()

# Sample data for faster processing
df = df.sample(frac=0.4, random_state=42)
df = df.reset_index(drop=True)
x_values = df.select_dtypes(include=['number'])

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = x_values.corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', ax=ax)
st.pyplot(fig)

# Drop unnecessary columns
df = df.drop(['Source IP', 'Source Port', 'Dest IP', 'Dest Port'], axis=1)

# Remove outliers
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

# Fill missing values
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

df = fill_missing_values(df)

# Encode target
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])

# One-hot encoding
categorical_cols = df.select_dtypes(exclude='number').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluation function
def evaluate(model_name, y_pred):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F2 Score': fbeta_score(y_test, y_pred, beta=2, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro')
    }

# Display results
results = [
    evaluate('RandomForestClassifier', y_pred_rf),
    evaluate('XGBClassifier', y_pred_xgb),
    evaluate('DecisionTreeClassifier', y_pred_dt)
]

st.subheader("Model Evaluation Results")
st.dataframe(pd.DataFrame(results))

# Predictions on test data
st.subheader("Example Predictions")
start_index = 12
num_rows = 5
example_data = X_test.iloc[start_index: start_index + num_rows]

pred_rf = rf_model.predict(example_data)
pred_xgb = xgb_classifier.predict(example_data)
true_labels = y_test.iloc[start_index: start_index + num_rows].values

st.write("Random Forest Predictions:", pred_rf)
st.write("XGBoost Predictions:", pred_xgb)
st.write("Actual Labels:", true_labels)

st.write("Interpreted Predictions (Random Forest):")
for pred in pred_rf:
    st.write("Normal Traffic" if pred == 0 else "DDoS Attack")

st.write("Interpreted Predictions (XGBoost):")
for pred in pred_xgb:
    st.write("Normal Traffic" if pred == 0 else "DDoS Attack")

# Save model
joblib.dump(rf_model, 'ddos_model.pkl')
