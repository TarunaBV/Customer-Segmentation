import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar navigation
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", ["Dashboard", "Visualization"])


# ================= DASHBOARD ================= #

if page == "Dashboard":

    st.set_page_config(
        page_title="Customer Segmentation App",
        page_icon="🤖",
        layout="wide"
    )

    # Reduce spacing
    st.markdown("""
    <style>
    .block-container{
        padding-top:1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown(
        "<h1 style='text-align:center;'>Customer Segmentation App</h1>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Layout (make center smaller)
    left_space, main_col, right_col = st.columns([0.2, 2, 1])

    with main_col:

        # Customer Insights
        st.markdown("""
        <h2 style='color:#4CAF50;'>Customer Insights Dashboard</h2>
        <p style='font-size:17px; color:gray;'>
        Discover meaningful customer groups using machine learning powered segmentation.
        This platform helps businesses analyze customer income and spending patterns
        to identify valuable customer segments and make smarter marketing decisions.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Prediction section
        st.subheader("🔍 Predict Customer Segment")

        customer_id = st.text_input("Enter Customer Id")

        income = st.number_input(
            "Annual income (k$)",
            min_value=0
        )

        spending_score = st.number_input(
            "Spending score (1-100)",
            min_value=1,
            max_value=100
        )

        # Load model
        model = pickle.load(open(
            r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\model.pkl",
            "rb"
        ))

        scaler = pickle.load(open(
            r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\scaler.pkl",
            "rb"
        ))

        input_data = np.array([[income, spending_score]])
        scaled_input = scaler.transform(input_data)

        if st.button("🚀 Predict Cluster", use_container_width=True):

            prediction = model.predict(scaled_input)

            st.session_state["user_point"] = scaled_input
            st.session_state["prediction"] = prediction[0]

            if prediction[0] == 0:
                st.success("Cluster 0 → Medium Income & Medium Spending")
            elif prediction[0] == 1:
                st.success("Cluster 1 → High Income & High Spending")
            elif prediction[0] == 2:
                st.success("Cluster 2 → Low Income & High Spending")
            elif prediction[0] == 3:
                st.success("Cluster 3 → High Income & Low Spending")
            elif prediction[0] == 4:
                st.success("Cluster 4 → Low Income & Low Spending")

            st.info(f"Customer {customer_id} belongs to Cluster {prediction[0]}")


    # RIGHT PANEL
    with right_col:

        st.markdown("### 📊 Customer Cluster Explanation")

        st.markdown("""
        **Cluster 0**  
        Medium Income & Medium Spending  

        **Cluster 1**  
        High Income & High Spending  

        **Cluster 2**  
        Low Income & High Spending  

        **Cluster 3**  
        High Income & Low Spending  

        **Cluster 4**  
        Low Income & Low Spending  
        """)



# ================= VISUALIZATION ================= #

elif page == "Visualization":

    st.set_page_config(
        page_title="Visualization",
        page_icon="📊",
        layout="centered"
    )

    st.title("Customer Segmentation Visualization")

    # Load data
    df = pd.read_excel(
        r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\data\Mall Customers.xlsx"
    )

    model = pickle.load(open(
        r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\model.pkl",
        "rb"
    ))

    scaler = pickle.load(open(
        r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\scaler.pkl",
        "rb"
    ))

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)

    fig, ax = plt.subplots()

    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=model.labels_, alpha=0.6)

    centers = model.cluster_centers_

    ax.scatter(
        centers[:,0],
        centers[:,1],
        marker='X',
        color='blue',
        s=250,
        label="Cluster Centers"
    )

    if "user_point" in st.session_state:

        user_point = st.session_state["user_point"]
        cluster = st.session_state["prediction"]

        ax.scatter(
            user_point[:,0],
            user_point[:,1],
            color='red',
            s=250,
            label=f"User (Cluster {cluster})"
        )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.legend()

    st.pyplot(fig)