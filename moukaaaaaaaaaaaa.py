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
    layout="wide"
)

# =========================
# USER FORM
# =========================
def check_user():

    if "user_valid" not in st.session_state:
        st.session_state.user_valid = False

    if not st.session_state.user_valid:

        st.title("👤 Informations Utilisateur")

        with st.form("user_form"):
            nom = st.text_input("Nom")
            prenom = st.text_input("Prénom")
            email = st.text_input("Email")

            submit = st.form_submit_button("Entrer")

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
                    st.error("⚠️ Remplir tous les champs")

        return False

    return True


# =========================
# DATA
# =========================
@st.cache_data
def load_data():

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

    X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

    X_opt = X[:, [0, 1]]

    model = sm.OLS(endog=y, exog=X_opt).fit()

    return model


# =========================
# MAIN
# =========================
if check_user():

    df = load_data()
    model = train_model(df)

    # SIDEBAR
    with st.sidebar:

        user = st.session_state.user_info
        st.success(f"👋 {user['prenom']} {user['nom']}")
        st.caption(user["email"])

        if st.button("🚪 Logout"):
            st.session_state.user_valid = False
            st.rerun()

    st.title("🚀 Startup Profit Prediction")

    col1, col2 = st.columns(2)

    with col1:

        rd_spend = st.number_input("💡 R&D Spend", 0.0, 500000.0, 100000.0)
        administration = st.number_input("📋 Administration", 0.0, 500000.0, 120000.0)
        marketing_spend = st.number_input("📢 Marketing Spend", 0.0, 500000.0, 200000.0)

        state = st.selectbox(
            "📍 State",
            options=["New York", "California", "Florida"]
        )

    with col2:

        if 'predictions' not in st.session_state:
            st.session_state.predictions = []

        new_input = np.array([[1.0, rd_spend]])

        if st.button("Predict"):

            prediction = model.predict(new_input)[0]

            st.session_state.predictions.append({
                'R&D Spend': rd_spend,
                'Administration': administration,
                'Marketing Spend': marketing_spend,
                'State': state,
                'Predicted Profit': prediction
            })

            st.success(f"Predicted Profit: ${prediction:,.2f}")

    # =========================
    # BUTTONS SECTION
    # =========================

    st.markdown("---")
    st.subheader("📊 Results")

    colA, colB = st.columns(2)

    show_all = colA.button("📄 Show All Predictions")
    show_last = colB.button("🎯 Final Result Only")

    # SHOW LAST RESULT
    if show_last and st.session_state.predictions:

        last = st.session_state.predictions[-1]

        st.metric("💰 Profit", f"${last['Predicted Profit']:,.2f}")
        st.write(last)

    # SHOW ALL
    if show_all and st.session_state.predictions:

        df_pred = pd.DataFrame(st.session_state.predictions)
        st.dataframe(df_pred, use_container_width=True)

    # CLEAR BUTTON
    if st.session_state.predictions:
        if st.button("🗑️ Clear All"):
            st.session_state.predictions = []
            st.rerun()
