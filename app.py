import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Risk App",
    layout="wide"
)

st.title("Supply Chain Risk & Delivery Prediction App")

st.write(
    "Predict supply chain risk classification and delivery time deviation "
    "based on input features."
)


# ---------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# LOAD MODELS / SCALERS / DATA
# ---------------------------------------------------------

@st.cache_resource
def load_assets():

    # Classification
    clf_model = joblib.load(
        BASE_DIR / "best_classification_model.pkl"
    )

    scaler_clf = joblib.load(
        BASE_DIR / "scaler_clf.pkl"
    )

    label_encoder = joblib.load(
        BASE_DIR / "label_encoder.pkl"
    )

    # Regression
    reg_model = joblib.load(
        BASE_DIR / "best_regression_model.pkl"
    )

    scaler_reg = joblib.load(
        BASE_DIR / "scaler_reg.pkl"
    )

    # Dataset
    data_sample = pd.read_csv(
        BASE_DIR / "dynamic_supply_chain_logistics_dataset.csv"
    )

    # Same leakage columns used during training
    leakage_columns = [
        "timestamp",
        "risk_classification",
        "delivery_time_deviation",
        "delay_probability",
        "disruption_likelihood_score",
        "eta_variation_hours",
    ]

    feature_columns = data_sample.drop(
        columns=leakage_columns
    ).columns.tolist()

    return (
        clf_model,
        scaler_clf,
        label_encoder,
        reg_model,
        scaler_reg,
        feature_columns,
        data_sample
    )


(
    clf_model,
    scaler_clf,
    label_encoder,
    reg_model,
    scaler_reg,
    feature_columns,
    raw_data_for_ranges
) = load_assets()


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.sidebar.header("Input Features")


def user_input_features():

    input_data = {}

    for col in feature_columns:

        min_val = raw_data_for_ranges[col].min()
        max_val = raw_data_for_ranges[col].max()
        mean_val = raw_data_for_ranges[col].mean()

        # Handle constant feature
        if min_val == max_val:

            display_min = (
                float(min_val * 0.9)
                if min_val != 0
                else -1.0
            )

            display_max = (
                float(max_val * 1.1)
                if max_val != 0
                else 1.0
            )

            default_val = float(min_val)

        else:

            display_min = float(min_val)
            display_max = float(max_val)
            default_val = float(mean_val)

        input_data[col] = st.sidebar.number_input(
            f"{col.replace('_', ' ').title()}",
            min_value=display_min,
            max_value=display_max,
            value=default_val
        )

    return pd.DataFrame(
        [input_data],
        columns=feature_columns
    )


input_df = user_input_features()


# ---------------------------------------------------------
# SHOW INPUT
# ---------------------------------------------------------

st.subheader("User Input Parameters")

st.dataframe(input_df, use_container_width=True)


# ---------------------------------------------------------
# PREDICTION BUTTONS
# ---------------------------------------------------------

st.sidebar.markdown("---")

classification_button = st.sidebar.button(
    "Predict Risk Classification"
)

regression_button = st.sidebar.button(
    "Predict Delivery Time"
)


# ---------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------

if classification_button:

    scaled_input_clf = scaler_clf.transform(
        input_df
    )

    clf_prediction_encoded = clf_model.predict(
        scaled_input_clf
    )

    clf_prediction = label_encoder.inverse_transform(
        clf_prediction_encoded
    )

    st.subheader("Risk Classification Result")

    st.success(
        f"Predicted Risk Classification: **{clf_prediction[0]}**"
    )

    # Probability
    if hasattr(clf_model, "predict_proba"):

        clf_proba = clf_model.predict_proba(
            scaled_input_clf
        )

        st.write("#### Prediction Probabilities")

        proba_df = pd.DataFrame(
            clf_proba,
            columns=label_encoder.classes_
        )

        st.dataframe(
            proba_df.style.highlight_max(axis=1),
            use_container_width=True
        )


# ---------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------

if regression_button:

    scaled_input_reg = scaler_reg.transform(
        input_df
    )

    reg_prediction = reg_model.predict(
        scaled_input_reg
    )

    predicted_deviation = reg_prediction[0]

    st.subheader("Delivery Time Prediction")

    st.success(
        f"Predicted Delivery Time Deviation: "
        f"**{predicted_deviation:.2f} hours**"
    )

    if predicted_deviation > 0:

        st.info(
            f"The predicted delivery deviation is "
            f"{predicted_deviation:.2f} hours."
        )

    elif predicted_deviation < 0:

        st.info(
            f"The predicted deviation is "
            f"{abs(predicted_deviation):.2f} hours earlier."
        )

    else:

        st.info(
            "The predicted delivery time deviation is approximately zero."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.write(
    "Note: This app is for demonstration purposes. "
    "Model performance may vary."
)
