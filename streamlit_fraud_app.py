
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Streamlit configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("AI-Powered Credit Card Fraud Detection")

# Upload CSV
uploaded_file = st.file_uploader("Upload creditcard.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Overview
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write(df['Class'].value_counts())
    st.bar_chart(df['Class'].value_counts())

    # Preprocessing
    st.subheader("Preprocessing Data")
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    df.drop(['Time'], axis=1, inplace=True)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    st.write("After SMOTE balancing:")
    st.write(pd.Series(y_res).value_counts())

    # Build Model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_res.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    with st.spinner("Training the model..."):
        model.fit(X_res, y_res, epochs=10, batch_size=2048, validation_split=0.2, verbose=0)
    st.success("Model training complete!")

    # Evaluate Model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC AUC
    roc_score = roc_auc_score(y_test, y_pred_prob)
    st.subheader("ROC AUC Score")
    st.metric("ROC AUC", f"{roc_score:.4f}")
