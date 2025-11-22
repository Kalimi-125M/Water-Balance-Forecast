

import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# ------------ CONFIG / CONSTANTS ------------

TARGET_COL = "distribution_input"

FEATURE_COLS = [
    "uphc",
    "dma",
    "mhh_cons",
    "mnhh_cons",
    "mean_air_temperature",
    "annual_sunshine",
    "maximum_temperature",
    "annual_precipitation",
    "minimum_air_temperature",
    "year_num",
    "months",
]

NUMERIC_FEATURES = [
    "uphc",
    "dma",
    "mhh_cons",
    "mnhh_cons",
    "mean_air_temperature",
    "annual_sunshine",
    "maximum_temperature",
    "annual_precipitation",
    "minimum_air_temperature",
    "year_num",
]

CATEGORICAL_FEATURES = ["months"]

MODEL_FILE = "linear_regression_distribution_input.joblib"


# ------------ HELPERS ------------

def load_data(uploaded_file):
    """Load CSV or Excel into a DataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def prepare_data(df):
    """
    Check required columns, clean 'years', create 'year_num',
    and drop rows with missing key predictors.
    """
    required_cols = [
        "uphc",
        "dma",
        "mhh_cons",
        "mnhh_cons",
        "mean_air_temperature",
        "annual_sunshine",
        "maximum_temperature",
        "annual_precipitation",
        "minimum_air_temperature",
        "years",
        "months",
        TARGET_COL,
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    df = df.copy()

    # Extract numeric year from values like "2024-25 Forecast"
    df["years_clean"] = df["years"].astype(str).str.extract(r"(\d{4})")
    df["years_clean"] = pd.to_numeric(df["years_clean"], errors="coerce")

    df.dropna(subset=["years_clean"], inplace=True)
    df["years_clean"] = df["years_clean"].astype(int)

    # Convert to datetime and numeric year
    df["years"] = pd.to_datetime(df["years_clean"], format="%Y", errors="coerce")
    df["year_num"] = df["years"].dt.year

    # Drop rows where key predictors are missing
    predictors = [
        "uphc",
        "dma",
        "mhh_cons",
        "mnhh_cons",
        "mean_air_temperature",
        "annual_sunshine",
        "maximum_temperature",
        "annual_precipitation",
        "minimum_air_temperature",
        "year_num",
        TARGET_COL,
    ]
    df.dropna(subset=predictors, inplace=True)

    return df


def make_pipeline():
    """Create the preprocessing + Linear Regression pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    return pipe


def train_new_model(df):
    """
    Train a new Linear Regression pipeline, save it with joblib,
    and return model, test split, predictions, and metrics.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Save model
    joblib.dump(pipe, MODEL_FILE)

    return pipe, X_test, y_test, y_pred, r2, mae


def load_saved_model():
    """Load a saved model from disk if available."""
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            return model
        except Exception as e:
            st.error(f"Error loading saved model: {e}")
            return None
    else:
        return None


def evaluate_existing_model(df, model):
    """
    Evaluate a loaded model on a fresh train/test split using
    the current uploaded dataset.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, X_test, y_test, y_pred, r2, mae


def input_ui(df):
    """UI for entering a single scenario to predict distribution_input."""
    st.subheader("Enter input values to predict distribution_input")

    col1, col2 = st.columns(2)

    with col1:
        uphc = st.number_input("UPHC", value=float(df["uphc"].median()))
        dma = st.number_input("DMA", value=float(df["dma"].median()))
        mhh_cons = st.number_input("MHH Cons", value=float(df["mhh_cons"].median()))
        mnhh_cons = st.number_input("MNH H Cons", value=float(df["mnhh_cons"].median()))
        year_num = st.number_input(
            "Year",
            min_value=1900,
            max_value=2100,
            value=int(df["year_num"].max()),
        )

    with col2:
        mean_air = st.number_input(
            "Mean Air Temperature",
            value=float(df["mean_air_temperature"].median()),
        )
        sun = st.number_input(
            "Annual Sunshine",
            value=float(df["annual_sunshine"].median()),
        )
        max_temp = st.number_input(
            "Maximum Temperature",
            value=float(df["maximum_temperature"].median()),
        )
        rain = st.number_input(
            "Annual Precipitation",
            value=float(df["annual_precipitation"].median()),
        )
        min_temp = st.number_input(
            "Minimum Air Temperature",
            value=float(df["minimum_air_temperature"].median()),
        )

    months_list = sorted(df["months"].astype(str).unique().tolist())
    month_choice = st.selectbox("Month", months_list)

    input_df = pd.DataFrame([{
        "uphc": uphc,
        "dma": dma,
        "mhh_cons": mhh_cons,
        "mnhh_cons": mnhh_cons,
        "mean_air_temperature": mean_air,
        "annual_sunshine": sun,
        "maximum_temperature": max_temp,
        "annual_precipitation": rain,
        "minimum_air_temperature": min_temp,
        "year_num": year_num,
        "months": month_choice,
    }])

    return input_df


# ------------ MAIN APP ------------

def main():
    st.title("Distribution Input Predictor – Linear Regression with Joblib")

    st.write(
        """
        Upload your dataset, choose whether to **train a new model** or **load a saved one**, 
        then predict **distribution_input** for new scenarios.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if not uploaded_file:
        st.info("Please upload a data file to continue.")
        return

    df_raw = load_data(uploaded_file)
    if df_raw is None:
        return

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    df = prepare_data(df_raw)
    if df is None or df.empty:
        st.error("Data preparation failed or resulted in an empty dataset.")
        return

    st.subheader("Prepared Data Preview")
    st.dataframe(df.head())

    model_option = st.radio(
        "Model option",
        ["Train new model", "Load saved model (if available)"],
        index=0,
    )

    with st.spinner("Setting up model..."):
        if model_option == "Load saved model (if available)":
            saved_model = load_saved_model()
            if saved_model is None:
                st.warning("No saved model found. Training a new model instead.")
                model, X_test, y_test, y_pred, r2, mae = train_new_model(df)
            else:
                model, X_test, y_test, y_pred, r2, mae = evaluate_existing_model(
                    df, saved_model
                )
        else:
            model, X_test, y_test, y_pred, r2, mae = train_new_model(df)

    st.subheader("Model Performance (Linear Regression)")
    st.write(f"R² on test set: **{r2:.3f}**")
    st.write(f"MAE on test set: **{mae:.3f}**")

    comparison_df = pd.DataFrame({
        "Actual_distribution_input": y_test.values,
        "Predicted_distribution_input": y_pred,
    })
    comparison_df["Error"] = (
        comparison_df["Actual_distribution_input"]
        - comparison_df["Predicted_distribution_input"]
    )
    comparison_df["Absolute_Error"] = comparison_df["Error"].abs()

    st.markdown("**Sample of actual vs predicted on test set:**")
    st.dataframe(comparison_df.head(20))

    # Let user download the trained model file
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            st.download_button(
                label="Download trained model (joblib)",
                data=f,
                file_name=MODEL_FILE,
                mime="application/octet-stream",
            )

    # Prediction UI
    input_df = input_ui(df)

    if st.button("Predict distribution_input"):
        pred_value = model.predict(input_df)[0]
        st.success(f"Predicted distribution_input: **{pred_value:.2f}**")


if __name__ == "__main__":
    main()
