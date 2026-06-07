import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# =========================
# Load Model and Encoders
# =========================

model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>

/* Pointer cursor for buttons */
.stButton > button {
    cursor: pointer !important;
}

/* Pointer cursor for dropdowns */
div[data-baseweb="select"] {
    cursor: pointer !important;
}

/* Pointer cursor for sliders */
.stSlider {
    cursor: pointer !important;
}

/* Pointer cursor for number inputs */
.stNumberInput {
    cursor: pointer !important;
}

/* Pointer cursor for labels */
label {
    cursor: pointer !important;
}

/* Professional Predict Button */
.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    cursor: pointer !important;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction")
st.markdown(
    "Enter customer details below to predict the probability of customer churn."
)
st.markdown("Enter customer details to predict the likelihood of churn.")

# Geography and Gender
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

# Sliders
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    tenure = st.slider(
        "Tenure",
        min_value=0,
        max_value=10,
        value=3
    )

with col3:
    num_of_products = st.slider(
        "Products",
        min_value=1,
        max_value=5,
        value=2
    )

# Numeric Inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

# Card and Activity Status
col1, col2 = st.columns(2)

with col1:
    has_cr_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

with col2:
    is_active_member = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
    )

# =========================
# Prediction
# =========================

if st.button("🔍 Predict Churn", use_container_width=True):

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Base Input Data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    # Geography Encoding
    geo_input = pd.DataFrame({
        "Geography": [geography]
    })

    geo_encoded = onehot_encoder_geo.transform(geo_input)

    # Convert sparse matrix to dense if needed
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out()
    )

    # Merge Features
    input_df = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Match training column order
    if hasattr(scaler, "feature_names_in_"):
        input_df = input_df.reindex(
            columns=scaler.feature_names_in_,
            fill_value=0
        )

    # Scale Input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    prediction_proba = float(prediction[0][0])

    # Results
    st.markdown("---")
    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )

    if prediction_proba > 0.5:
        st.error(
            f"⚠️ Customer is likely to churn "
            f"({prediction_proba:.2%} probability)"
        )
    else:
        st.success(
            f"✅ Customer is unlikely to churn "
            f"({1 - prediction_proba:.2%} confidence)"
        )

    # Optional probability bar
    st.progress(float(prediction_proba))