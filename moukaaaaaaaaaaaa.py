import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# USER FORM (Nom / Prenom / Email)
# =========================
def check_user():

    if "user_valid" not in st.session_state:
        st.session_state.user_valid = False

    if not st.session_state.user_valid:
        st.markdown("### 👤 Informations Utilisateur")
        st.markdown("---")

        with st.form("user_form"):
            nom = st.text_input("Nom")
            prenom = st.text_input("Prénom")
            email = st.text_input("Email")

            submit = st.form_submit_button("Entrer dans l'application")

            if submit:
                if nom and prenom and email:
                    st.session_state.user_valid = True
                    st.session_state.user_info = {
                        "nom": nom,
                        "prenom": prenom,
                        "email": email
                    }
                    st.rerun()
                else:
                    st.error("⚠️ Veuillez remplir tous les champs")

        return False

    return True


# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('50_Startups.csv')
        return df
    except FileNotFoundError:
        st.warning("CSV file not found. Using sample data.")
        data = {
            'R&D Spend': [165349.2, 162597.7, 153441.51, 144372.41, 142107.34],
            'Administration': [136897.8, 151377.59, 101145.55, 118671.85, 91391.77],
            'Marketing Spend': [471784.1, 443898.53, 407934.54, 383199.62, 366168.42],
            'State': ['New York', 'California', 'Florida', 'New York', 'Florida'],
            'Profit': [192261.83, 191792.06, 191050.39, 182901.99, 166187.94]
        }
        return pd.DataFrame(data)


@st.cache_resource
def train_model(df):

    data = df.copy()

    le = LabelEncoder()
    data['State'] = le.fit_transform(data['State'])

    X = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']].values
    y = data['Profit'].values

    # Add constant
    X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

    # Final model uses const + R&D Spend
    X_opt = X[:, [0, 1]]

    model = sm.OLS(endog=y, exog=X_opt).fit()

    return model


# =========================
# VALIDATION
# =========================
def validate_inputs(rd_spend, admin, marketing):

    errors = []

    if rd_spend < 0:
        errors.append("❌ R&D Spend cannot be negative")

    if admin < 0:
        errors.append("❌ Administration cannot be negative")

    if marketing < 0:
        errors.append("❌ Marketing Spend cannot be negative")

    return errors


# =========================
# MAIN APP
# =========================
if check_user():

    try:

        df = load_data()
        model = train_model(df)

        # SIDEBAR
        with st.sidebar:

            if "user_info" in st.session_state:
                user = st.session_state.user_info
                st.success(f"👋 Bienvenue {user['prenom']} {user['nom']}")
                st.caption(user["email"])

            st.markdown("---")

            st.header("📊 Model Stats")
            st.metric("R²", f"{model.rsquared:.4f}")
            st.metric("Observations", len(df))

            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.user_valid = False
                st.rerun()

        # MAIN TITLE
        st.title("🚀 Startup Profit Prediction System")
        st.markdown("Predict startup profit based on spending and location")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📝 Input Variables")

            rd_spend = st.number_input("💡 R&D Spend", 0.0, 500000.0, 100000.0, 1000.0)
            administration = st.number_input("📋 Administration", 0.0, 500000.0, 120000.0, 1000.0)
            marketing_spend = st.number_input("📢 Marketing Spend", 0.0, 500000.0, 200000.0, 1000.0)

            state = st.selectbox(
                "📍 State",
                options=["New York", "California", "Florida"]
            )

        with col2:

            st.subheader("🎯 Prediction")

            validation_errors = validate_inputs(
                rd_spend, administration, marketing_spend
            )

            if validation_errors:
                for err in validation_errors:
                    st.error(err)

            new_input = np.array([[1.0, rd_spend]])

            if 'predictions' not in st.session_state:
                st.session_state.predictions = []

            if st.button("Predict", disabled=bool(validation_errors)):

                prediction = model.predict(new_input)[0]

                st.session_state.predictions.append({
                    'R&D Spend': rd_spend,
                    'Administration': administration,
                    'Marketing Spend': marketing_spend,
                    'State': state,
                    'Predicted Profit': prediction
                })

        # RESULTS
        st.markdown("---")

        if st.session_state.predictions:

            last = st.session_state.predictions[-1]

            st.subheader("📈 Latest Prediction")

            colA, colB, colC = st.columns(3)

            colA.metric("💰 Profit", f"${last['Predicted Profit']:,.2f}")
            colB.metric("💡 R&D", f"${last['R&D Spend']:,.2f}")
            colC.metric("📍 State", last['State'])

            df_pred = pd.DataFrame(st.session_state.predictions)

            st.dataframe(df_pred, use_container_width=True)

            if st.button("🗑️ Clear"):
                st.session_state.predictions = []
                st.rerun()

        # DATASET
        st.markdown("---")

        st.subheader("📊 Dataset Overview")

        col3, col4, col5, col6 = st.columns(4)

        col3.metric("Startups", len(df))
        col4.metric("Avg Profit", f"${df['Profit'].mean():,.0f}")
        col5.metric("Min Profit", f"${df['Profit'].min():,.0f}")
        col6.metric("Max Profit", f"${df['Profit'].max():,.0f}")

        with st.expander("View Dataset"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(str(e))
