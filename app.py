import streamlit as st
import pandas as pd
import xgboost as xgb

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# --------------------------------------------------
# LOAD TRAINED XGBOOST MODEL
# --------------------------------------------------

model = xgb.Booster()
model.load_model("churn_model.json")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details to predict the probability "
    "of customer churn."
)

st.divider()

# --------------------------------------------------
# CUSTOMER INPUTS
# --------------------------------------------------

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

has_cr_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

# --------------------------------------------------
# CONVERT CATEGORICAL INPUTS
# --------------------------------------------------

# Gender
gender_value = 1 if gender == "Male" else 0

# Credit Card
has_cr_card_value = 1 if has_cr_card == "Yes" else 0

# Active Member
is_active_member = 1 if active_member == "Yes" else 0

# Geography One-Hot Encoding
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔍 Predict Churn"):

    # --------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_value],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card_value],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
        "Geography_Germany": [geography_germany],
        "Geography_Spain": [geography_spain]
    })

    # --------------------------------------------------
    # ENSURE CORRECT COLUMN ORDER
    # --------------------------------------------------

    input_data = input_data[
        [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_Germany",
            "Geography_Spain"
        ]
    ]

    # --------------------------------------------------
    # CREATE XGBOOST DMATRIX
    # --------------------------------------------------

    dmatrix = xgb.DMatrix(input_data)

    # --------------------------------------------------
    # PREDICT CHURN PROBABILITY
    # --------------------------------------------------

    probability = model.predict(dmatrix)[0]

    # Convert to percentage
    churn_probability = probability * 100

    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )

    # --------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------

    if probability >= 0.70:

        risk = "HIGH RISK"

        st.error(
            "🔴 High Risk: Customer is likely to churn."
        )

        st.write(
            "Recommended Action: Contact the customer "
            "and offer a retention plan."
        )

    elif probability >= 0.40:

        risk = "MEDIUM RISK"

        st.warning(
            "🟡 Medium Risk: Customer may churn."
        )

        st.write(
            "Recommended Action: Monitor the customer "
            "and consider engagement offers."
        )

    else:

        risk = "LOW RISK"

        st.success(
            "🟢 Low Risk: Customer is unlikely to churn."
        )

        st.write(
            "Recommended Action: Continue normal "
            "customer engagement."
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    st.write(
        f"### Risk Level: **{risk}**"
    )