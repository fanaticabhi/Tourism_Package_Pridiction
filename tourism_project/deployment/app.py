import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(
    repo_id="abhishek1504/wellness-tourism-model",
    filename="wellness_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism App")
st.write("""
This application predicts whether a customer will take a tourism package
based on their demographic and interaction details.
""")

# ----------- User Inputs -----------
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)

ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=1)

Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input("Monthly Income", min_value=10000, max_value=500000, value=30000)

# ----------- Assemble Input Data -----------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# ----------- Prediction -----------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success("Customer WILL purchase the package")
    else:
        st.error("Customer WILL NOT purchase the package")
