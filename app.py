import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #B3CDE0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "laptop_price_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "laptop_data_cleaned_final.csv")

COMPANIES = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP',
             'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer',
             'Samsung', 'Toshiba', 'Vaio', 'Xiaomi']
TYPES = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
CPUS = ['AMD Processor', 'Intel Core i3', 'Intel Core i5', 'Intel Core i7',
        'Other Intel Processor', 'Other Processor']
GPUS = ['AMD', 'Intel', 'Nvidia']
OS_LIST = ['Windows', 'Mac', 'Others/No OS/Linux']
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown(
    """
    <div style="
        background-color:#E6DFF7;
        padding:22px;
        border-radius:14px;
        text-align:center;
        margin-bottom:20px;
    ">
        <h2 style="color:#4B2E83; font-size:26px; margin:0;">
            💻 SmartPrice AI
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "Navigate",
    ["📘 Project Overview", "🔮 Predict Price", "📊 Dataset Insights", "🛠️ Tech Stack", "👩‍💻 About Me"]
)


# ============================================================
# PAGE 1 — PROJECT OVERVIEW
# ============================================================
if page == "📘 Project Overview":

    st.markdown(
        """
        <div style="
            background-color:#0E4D64;
            padding:25px;
            border-radius:16px;
            text-align:center;
            margin-bottom:15px;
        ">
            <h1 style="color:white; font-size:44px; margin:0;">
                💻 SmartPrice AI
            </h1>
        </div>
        <div style="
            background-color:#1560BD;
            padding:30px;
            border-radius:16px;
            text-align:center;
            margin-bottom:30px;
        ">
            <h2 style="color:white; font-size:30px; margin-bottom:5px;">
                A Laptop Value Prediction Engine
            </h2>
            <p style="color:#D6EFFF; font-size:18px; margin:0;">
                Estimate a laptop's fair market price from its specifications
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This app predicts the **price of a laptop** based on its specifications
    (brand, type, RAM, storage, GPU, screen quality, and more) using a
    machine learning model trained on ~6,000 real-world laptop listings.
    """)

    st.subheader("🤖 Models Compared")
    st.markdown(
        """
        <div style="
            background-color:#FFFFFF;
            border:1px solid #D6E0EA;
            border-radius:14px;
            padding:25px;
            margin-bottom:20px;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);
            color : "black";
        ">
            <table style="width:100%; border-collapse:collapse; text-align:center;">
                <tr style="background-color:#0E4D64; color:white;">
                    <th style="padding:10px;">Model</th>
                    <th style="padding:10px;">R² Score</th>
                    <th style="padding:10px;">MAE</th>
                </tr>
                <tr>
                    <td style="padding: 8px;color: black;">Linear Regression</td>
                    <td style="padding:8px;color: black;">0.749</td>
                    <td style="padding:8px;color: black;">₹7,211</td>
                </tr>
                <tr style="background-color:#F5F7FA;color: black;">
                    <td style="padding:8px;">KNN</td>
                    <td style="padding:8px;">0.835</td>
                    <td style="padding:8px;">₹10,277</td>
                </tr>
                <tr>
                    <td style="padding:8px;color: black;">SVM (SVR)</td>
                    <td style="padding:8px;color: black;">0.926</td>
                    <td style="padding:8px;color: black;">₹5,766</td>
                </tr>
                <tr style="background-color:#D6EFFF; font-weight:bold;">
                    <td style="padding:8px;color: black;">Random Forest</td>
                    <td style="padding:8px;color: black;">0.953</td>
                    <td style="padding:8px;color: black;">₹5,477</td>
                </tr>
            </table>
            <p style="margin-top:15px; margin-bottom:0; color:#333;">
                <b>Random Forest</b> was selected as the final model — highest R²,
                lowest error on the held-out test set.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Open **🔮 Predict Price** from the sidebar.
    2. Enter the laptop's specifications (brand, RAM, storage, display, etc.).
    3. Click **Predict Price** — get an instant price estimate 🎈
    4. Check **📊 Dataset Insights** to explore the data behind the model.
    """)

    st.info("Built as a complete ML workflow: EDA → Preprocessing → Model Training → Model Selection → Deployment (Streamlit).")


# ============================================================
# PAGE 2 — PREDICTION INTERFACE
# ============================================================
elif page == "🔮 Predict Price":
    st.title("🔮 Predict Laptop Price")
    st.markdown("Fill in the specifications below and click **Predict Price**.")

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load `laptop_price_model.pkl`. Make sure the file is in the "
            f"same folder as `app.py`.\n\nDetails: {e}"
        )
        st.stop()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Brand, Type & Performance")
        company = st.selectbox("Company", COMPANIES)
        typename = st.selectbox("Laptop Type", TYPES)
        cpu = st.selectbox("CPU", CPUS)
        gpu = st.selectbox("GPU Brand", GPUS)
        os_choice = st.selectbox("Operating System", OS_LIST)
        ram = st.selectbox("RAM (GB)", RAM_OPTIONS, index=3)
        weight = st.slider("Weight (kg)", 0.9, 4.7, 2.0, 0.1)

    with col2:
        st.subheader("Storage & Display")
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 512, 1000, 2000], index=0)
        ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 180, 256, 512, 1000], index=4)
        flash_storage = st.selectbox("Flash Storage (GB)", [0, 16, 32, 64, 128, 256], index=0)
        hybrid = st.selectbox("Hybrid Storage (GB)", [0, 500, 1000, 2000], index=0)
        ppi = st.slider("PPI (Pixels Per Inch)", 85, 436, 141)
        touchscreen = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS Panel")

    st.divider()

    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "Company": company,
            "TypeName": typename,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": int(touchscreen),
            "Ips": int(ips),
            "ppi": ppi,
            "Cpu_name": cpu,
            "HDD": float(hdd),
            "SSD": float(ssd),
            "Hybrid": float(hybrid),
            "Flash_Storage": float(flash_storage),
            "Gpu_brand": gpu,
            "os": os_choice,
        }])

        log_price = model.predict(input_df)[0]
        price = np.exp(log_price)

        st.markdown(
            f"""
            <div style="
                background-color:#0E4D64;
                padding:45px;
                border-radius:18px;
                text-align:center;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <p style="color:#D6EFFF; font-size:20px; margin:0;">Estimated Price</p>
                <h1 style="color:white; font-size:56px; margin:10px 0 0 0;">
                    ₹{price:,.2f}
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()
        st.caption("Note: This is a statistical estimate based on historical listing data, not a guaranteed market price.")

        with st.expander("See input summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))


# ============================================================
# PAGE 3 — DATASET INSIGHTS
# ============================================================
elif page == "📊 Dataset Insights":
    st.title("📊 Dataset Insights")
    st.markdown("Explore key relationships in the training data.")

    try:
        df = load_data()
    except Exception as e:
        st.error(
            f"Could not load `laptop_data_cleaned_final.csv`. Make sure the file is "
            f"in the same folder as `app.py`.\n\nDetails: {e}"
        )
        st.stop()

    sns.set_style("whitegrid")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RAM vs Price")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(x="Ram", y="Price", data=df, alpha=0.5, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("SSD vs Price")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(x="SSD", y="Price", data=df, alpha=0.5, color="green", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Average Price by Brand")
        avg_price = df.groupby("Company")["Price"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=avg_price.values, y=avg_price.index, hue=avg_price.index, legend=False, palette="viridis", ax=ax)
        ax.set_xlabel("Average Price")
        st.pyplot(fig)

    with col4:
        st.subheader("CPU Brand Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(y="Cpu_name", data=df, order=df["Cpu_name"].value_counts().index, hue="Cpu_name", legend=False, palette="crest", ax=ax)
        ax.set_xlabel("Count")
        st.pyplot(fig)


# ============================================================
# PAGE 4 — TECH STACK
# ============================================================
elif page == "🛠️ Tech Stack":
    st.title("🛠️ Tech Stack")
    st.markdown("Tools and libraries used to build this project.")
    st.divider()

    tech = [
        ("🐍", "Python", "Core programming language"),
        ("🐼", "Pandas", "Data loading & manipulation"),
        ("🔢", "NumPy", "Numerical computations"),
        ("🤖", "Scikit-learn", "ML models & preprocessing pipeline"),
        ("📈", "Matplotlib", "Data visualization"),
        ("🎈", "Streamlit", "Web app interface"),
        ("📦", "Pickle / Joblib", "Model serialization & export"),
    ]

    card_style = """
        <div style="
            background-color:#F0F4F8;
            border:1px solid #D6E0EA;
            border-radius:14px;
            padding:25px;
            text-align:center;
            height:150px;
            margin-bottom:15px;
        ">
            <div style="font-size:36px;">{icon}</div>
            <h4 style="margin:8px 0 4px 0; color:#0E4D64;">{name}</h4>
            <p style="font-size:13px; color:#555; margin:0;">{desc}</p>
        </div>
    """

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(card_style.format(icon=icon, name=name, desc=desc), unsafe_allow_html=True)


# ============================================================
# PAGE 5 — ABOUT ME
# ============================================================
elif page == "👩‍💻 About Me":
    st.markdown(
        """
        <div style="
            background-color:#F8BBD0;
            padding:55px;
            border-radius:18px;
            text-align:center;
            margin-top:30px;
            margin-bottom:20px;
        ">
            <h1 style="color:#880E4F; font-family:Georgia, serif; font-size:52px; margin:0;">
                Developed by Jigyasa Jindal
            </h1>
        </div>
        <div style="
            background-color:#FCE4EC;
            padding:20px;
            border-radius:14px;
            text-align:center;
        ">
            <a href="https://www.linkedin.com/in/jigyasa-jindal-4752b7372" target="_blank"
               style="color:#AD1457; font-size:18px; font-weight:600; text-decoration:underline;">
                🔗 Connect on LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

