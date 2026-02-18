import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="💰",
    layout="wide"
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
# CHECK USER BEFORE LOADING APP
# =========================
if check_user():

    # =========================
    # APP TITLE
    # =========================
    st.title("💰 Automatic Backward Elimination (OLS)")
    st.markdown("---")

    if "user_info" in st.session_state:
        user = st.session_state.user_info
        st.sidebar.success(f"👋 Bienvenue {user['prenom']} {user['nom']}")
        st.sidebar.caption(user["email"])
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.user_valid = False
            st.rerun()

    # =========================
    # TRAIN MODEL
    # =========================
    @st.cache_resource
    def train_model():

        np.random.seed(42)
        n = 50

        rd = np.random.uniform(0, 200000, n)
        admin = np.random.uniform(50000, 150000, n)
        marketing = np.random.uniform(0, 300000, n)
        state = np.random.choice([0, 1, 2], n)  # CA, NY, FL

        profit = (
            49000
            + 0.85 * rd
            + 0.02 * marketing
            + np.random.normal(0, 10000, n)
        )

        data = pd.DataFrame({
            "R&D Spend": rd,
            "Administration": admin,
            "Marketing Spend": marketing,
            "State_NY": (state == 1).astype(int),
            "State_FL": (state == 2).astype(int),
            "Profit": profit
        })

        X = data.drop("Profit", axis=1)
        y = data["Profit"]

        X = sm.add_constant(X)

        SL = 0.05
        all_steps = []
        step = 1

        while True:
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues
            max_p = p_values.max()

            all_steps.append({
                "step": step,
                "variables": list(X.columns),
                "summary": model.summary().as_text()
            })

            if max_p > SL:
                feature_to_remove = p_values.idxmax()
                X = X.drop(columns=[feature_to_remove])
                step += 1
            else:
                break

        results = {
            "model": model,
            "selected_columns": X.columns,
            "all_steps": all_steps,
            "final_summary": model.summary().as_text()
        }

        return results

    BACKWARD_ELIMINATION_RESULTS = train_model()
    model = BACKWARD_ELIMINATION_RESULTS["model"]
    selected_columns = BACKWARD_ELIMINATION_RESULTS["selected_columns"]

    # =========================
    # USER INPUT
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        rd_input = st.number_input("R&D Spend", 0.0, 200000.0, 50000.0)
        admin_input = st.number_input("Administration", 0.0, 200000.0, 120000.0)
        marketing_input = st.number_input("Marketing Spend", 0.0, 300000.0, 200000.0)

        state_input = st.selectbox(
            "State",
            ["California", "New York", "Florida"]
        )

    with col2:
        predict_button = st.button("🎯 Predict Profit", type="primary", use_container_width=True)
        show_all_button = st.button("📊 Tous les résultats", use_container_width=True)
        show_last_button = st.button("🎯 Résultat final", use_container_width=True)

    # =========================
    # PREDICTION
    # =========================
    if predict_button:

        input_data = pd.DataFrame({
            "R&D Spend": [rd_input],
            "Administration": [admin_input],
            "Marketing Spend": [marketing_input],
            "State_NY": [1 if state_input == "New York" else 0],
            "State_FL": [1 if state_input == "Florida" else 0]
        })

        input_data = sm.add_constant(input_data, has_constant="add")
        input_data = input_data[selected_columns]
        prediction = model.predict(input_data)

        st.success("✅ Prediction Successful")
        st.metric("💵 Predicted Profit", f"${prediction.iloc[0]:,.2f}")

    # =========================
    # SHOW ALL BACKWARD STEPS
    # =========================
    if show_all_button:
        st.markdown("## 📊 Toutes les étapes du Backward Elimination")
        all_steps = BACKWARD_ELIMINATION_RESULTS["all_steps"]

        st.info(f"Nombre total d'étapes : *{len(all_steps)}*")

        for step in all_steps:
            with st.expander(
                f"Étape {step['step']} — Variables: {step['variables']}",
                expanded=(step["step"] == 1)
            ):
                st.code(step["summary"])

    # =========================
    # SHOW FINAL MODEL ONLY
    # =========================
    if show_last_button:
        st.markdown("## 🎯 Modèle Final")

        st.code(BACKWARD_ELIMINATION_RESULTS["final_summary"])

        coef = model.params
        equation_terms = []

        for name, value in coef.items():
            if name == "const":
                equation_terms.append(f"{value:.2f}")
            else:
                equation_terms.append(f"{value:.4f} × {name}")

        equation = "Profit = " + " + ".join(equation_terms)

        st.markdown("### 📐 Équation du modèle final")
        st.latex(equation)

        st.success(
            f"R² = {model.rsquared:.3f} | Adjusted R² = {model.rsquared_adj:.3f}"
        )
