import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Customer Cluster Predictor", page_icon="🤖")

model = pickle.load(open("D:\GITHUB\Customer Segmentation\model\model.pkl", "rb"))
scaler = pickle.load(open("D:\GITHUB\Customer Segmentation\model\scaler.pkl", "rb"))

st.title("Customer Segmentation App")

st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Visualization"])

if page == "Home":
    st.write("Description add madu!")

elif page == "Predict":
    customer_id = st.text_input("Enter Customer Id : ")
    income = st.number_input("Annual income (k$)", min_value=0)
    spending_score = st.number_input("Spending score (1-100)", min_value=1, max_value=100)

    input_data = np.array([[income, spending_score]])
    scaled_input = scaler.transform(input_data)

    if st.button("Predict the Cluter"):
        prediction = model.predict(scaled_input)
        if prediction[0] == 0:
            st.write(f"Customer {customer_id} have Medium Income & have Medium Spending History.")
        elif prediction[0] == 1:
            st.write(f"Customer {customer_id} have High Income & have High Spending History.")
        elif prediction[0] == 2:
            st.write(f"Customer {customer_id} have Low Income & have High Spending History.")
        elif prediction[0] == 3:
            st.write(f"Customer {customer_id} have High Income & Low Spending History.")
        elif prediction[0] == 4:
            st.write(f"Customer {customer_id} have Low Income & have Low Spending History.")

        st.success(f"Customer belongs to Cluster {prediction[0]}")

elif page == "Visualization":
    st.write("Visualization add madu!")
