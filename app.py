import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/coupon_success_model.pkl")

st.title("Coupon Success Prediction")

st.write("Predict the likelihood of a coupon being successful before launch.")

# User inputs
facevalue = st.slider("Face Value ($)", 0.5, 5.0, 1.5)
mthsactive = st.slider("Months Active", 1, 6, 3)
distquantity = st.number_input("Distribution Quantity", 10000, 500000, 100000)
media = st.selectbox("Media", ["Digital", "Print", "InStore"])
brand = st.selectbox("Brand", ["BrandA", "BrandB", "BrandC"])
price = st.number_input("Product Selling Price ($)", 1.0, 10.0, 4.99)
budget = st.number_input("Budget Dollars", 50000, 500000, 150000)
forecastqty = st.number_input("Forecasted Quantity", 5000, 50000, 20000)

# Predict
if st.button("Predict Success"):
    input_df = pd.DataFrame(
        {
            "FACEVALUE": [facevalue],
            "MTHSACTIVE": [mthsactive],
            "DISTQUANTITY": [distquantity],
            "MEDIA": [media],
            "BRAND": [brand],
            "PRODUCTSELLINGPRICE": [price],
            "BUDGET_DOLLARS": [budget],
            "FORECASTEDQTY": [forecastqty],
        }
    )

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Success Probability:** {prob:.2%}")

    if prob >= 0.7:
        st.success("Low risk – likely to be successful")
    elif prob >= 0.4:
        st.warning("Medium risk – monitor closely")
    else:
        st.error("High risk – likely to underperform")
