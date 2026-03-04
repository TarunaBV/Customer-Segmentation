import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Customer Cluster Predictor", page_icon="🤖")

model = pickle.load(open(r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\model\scaler.pkl", "rb"))

st.markdown(
    "<h1 style='text-align: center;'>Customer Segmentation App</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Visualization"])

if page == "Home":
    st.markdown("""
        <style>
        .block-container {
            max-width: 90%;
            padding-top: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)


    # HERO SECTION (Insights + Cluster Description side by side)
    col1, col2 = st.columns([2,2])

    with col1:
        st.markdown("""
        <div style='padding:40px 10px;'>
            <h1 style='font-size:42px; color:#4CAF50;'>
            Customer Insights Dashboard
            </h1>
            <p style='font-size:20px; color:gray;'>
            Discover meaningful customer groups using machine learning powered segmentation.
            This platform helps businesses analyze customer income and spending patterns
            to identify valuable customer segments and make smarter marketing decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
          <div style='padding:30px; border-radius:12px; background-color:#f5f5f5; color:#1f1f1f;'>

    <h2 style='color:#1f1f1f;'>Customer Segments</h2>


    <p style='font-size:20px;'>
    <b>Cluster 0</b> → Medium Income & Medium Spending
    </p>

    <p style='font-size:20px;'>
    <b>Cluster 1</b> → High Income & High Spending
    </p>

    <p style='font-size:20px;'>
    <b>Cluster 2</b> → Low Income & High Spending
    </p>

    <p style='font-size:20px;'>
    <b>Cluster 3</b> → High Income & Low Spending
    </p>

    <p style='font-size:20px;'>
    <b>Cluster 4</b> → Low Income & Low Spending
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # FEATURE CARDS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='padding:20px; border-radius:10px; background-color:#f5f5f5; color:#1f1f1f;'>
            <h3>📊 Data Analysis</h3>
            <p>
            Analyze annual income and spending patterns 
            to discover hidden customer segments.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='padding:20px; border-radius:10px; background-color:#f5f5f5;color:#1f1f1f;'>
            <h3>🤖 ML Powered</h3>
            <p>
            Uses K-Means clustering algorithm to automatically 
            group customers with similar behaviors.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='padding:20px; border-radius:10px; background-color:#f5f5f5;color:#1f1f1f;'>
            <h3>🎯 Business Impact</h3>
            <p>
            Helps businesses design targeted marketing strategies 
            and improve customer engagement.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CALL TO ACTION
    st.markdown("""
    <div style='text-align:center; padding:20px;'>
        <h2>Ready to Explore Insights?</h2>
        <p style='color:gray; font-size:18px;'>
        Use the navigation menu to predict customer segments 
        and visualize clustering results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # FOOTER
    st.markdown("""
    <div style='text-align:center; color:gray; padding:15px; font-size:14px;'>
    © 2026 Customer Segmentation Project | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

elif page == "Predict":

    customer_id = st.text_input("Enter Customer Id : ")
    income = st.number_input("Annual income (k$)", min_value=0)
    spending_score = st.number_input("Spending score (1-100)", min_value=1, max_value=100)

    input_data = np.array([[income, spending_score]])
    scaled_input = scaler.transform(input_data)

    if st.button("Predict the Cluster"):

        prediction = model.predict(scaled_input)

        # Save user data
        st.session_state["user_point"] = scaled_input
        st.session_state["prediction"] = prediction[0]

        if prediction[0] == 0:
            st.write(f"Customer {customer_id} have Medium Income & Medium Spending.")
        elif prediction[0] == 1:
            st.write(f"Customer {customer_id} have High Income & High Spending.")
        elif prediction[0] == 2:
            st.write(f"Customer {customer_id} have Low Income & High Spending.")
        elif prediction[0] == 3:
            st.write(f"Customer {customer_id} have High Income & Low Spending.")
        elif prediction[0] == 4:
            st.write(f"Customer {customer_id} have Low Income & Low Spending.")

        st.success(f"Customer belongs to Cluster {prediction[0]}")

elif page == "Visualization":

    st.subheader("📊 Customer Segmentation Visualization")

    df = pd.read_excel(r"C:\Users\vaish\OneDrive\Documents\GitHub\Customer-Segmentation\data\Mall Customers.xlsx")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)

    fig, ax = plt.subplots()

    # Plot all customers
    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=model.labels_, alpha=0.5)

    # Plot cluster centers
    centers = model.cluster_centers_
    ax.scatter(centers[:,0], centers[:,1], marker='X', color = 'blue', s=250, label="Cluster Centers")

    # Plot user point if exists
    if "user_point" in st.session_state:
        user_point = st.session_state["user_point"]
        cluster = st.session_state["prediction"]

        ax.scatter(user_point[:,0], user_point[:,1],
                   color='red', s=250, label=f"User (Cluster {cluster})")

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.legend()

    st.pyplot(fig)