"""
Streamlit app for the "Visit with Us" Wellness Tourism Package predictor.

Loads the model committed to this folder by the training pipeline, collects
customer details through a form, and predicts whether the customer is
likely to purchase the package.
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Wellness Tourism Package Predictor")
st.write(
    "Enter the customer's details below to predict whether they are likely "
    "to purchase the new Wellness Tourism Package."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found at tourism_project/deployment/model.joblib. "
        "Run the training pipeline first so it commits a model to the repo."
    )
    st.stop()

model = load_model()

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=37)
    city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation", options=["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", options=["Female", "Male"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    designation = st.selectbox(
        "Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=200000, value=23000, step=500
    )

with col2:
    number_of_persons = st.number_input(
        "Number Of Persons Visiting", min_value=1, max_value=10, value=3
    )
    number_of_children = st.number_input(
        "Number Of Children Visiting (below age 5)", min_value=0, max_value=10, value=1
    )
    number_of_trips = st.number_input(
        "Number Of Trips (per year)", min_value=0, max_value=30, value=3
    )
    preferred_star = st.selectbox("Preferred Property Star", options=[3.0, 4.0, 5.0], index=0)
    passport = st.selectbox("Holds Passport?", options=["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", options=["Yes", "No"])

st.header("Sales Interaction Details")
col3, col4 = st.columns(2)

with col3:
    type_of_contact = st.selectbox(
        "Type of Contact", options=["Self Enquiry", "Company Invited"]
    )
    product_pitched = st.selectbox(
        "Product Pitched", options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

with col4:
    duration_of_pitch = st.number_input(
        "Duration Of Pitch (minutes)", min_value=1, max_value=180, value=15
    )
    number_of_followups = st.number_input(
        "Number Of Followups", min_value=0, max_value=10, value=4
    )
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": type_of_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": number_of_persons,
                "NumberOfFollowups": number_of_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": number_of_trips,
                "Passport": 1 if passport == "Yes" else 0,
                "PitchSatisfactionScore": pitch_satisfaction,
                "OwnCar": 1 if own_car == "Yes" else 0,
                "NumberOfChildrenVisiting": number_of_children,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
    else:
        st.info(f"Unlikely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
