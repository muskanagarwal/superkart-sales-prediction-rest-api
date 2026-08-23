import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product & store features
product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.66,
    help="Weight of the product (numerical value)",
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.068,
    help="Ratio of the allocated display area of each product to the total display area of all the products in a store",
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=116.7,
    help="Maximum retail price of each product (numerical value)",
)

store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"],
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
)

store_age_years = st.number_input(
    "Store Age (Years)",
    min_value=0,
    value=17,
    help="Age of the store"
)

product_type_category = st.selectbox(
    "Product Type Category",
    ["Perishables", "Non Perishables"]
)

product_id_char = st.selectbox(
    "Product Id Char",
    ["FD", "NC", "DR"]
)

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Store_Age_Years': store_age_years,
    'Product_Type_Category': product_type_category,
    'Product_Id_char': product_id_char
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Sales']
        st.success(f"Predicted Sales: {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predict/batch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
