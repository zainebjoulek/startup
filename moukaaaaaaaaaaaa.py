import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import smtplib
import random
import datetime
import os
import jwt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(_file_)), ".env"))

def _secret(key: str, fallback: str = "") -> str:
    """Read from Streamlit secrets (Cloud) or .env (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, fallback)

SMTP_EMAIL    = _secret("SMTP_EMAIL")
SMTP_PASSWORD = _secret("SMTP_PASSWORD")
JWT_SECRET    = _secret("JWT_SECRET", "default-secret-change-me")

# =========================
# EMAIL OTP SENDER
# =========================
def send_otp_email(to_email: str, otp_code: str, nom: str, prenom: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "🔐 Code de vérification – Startup Profit Predictor"
    msg["From"]    = SMTP_EMAIL
    msg["To"]      = to_email

    html_body = f"""
    <html>
    <body style="font-family:Arial,sans-serif;padding:30px;">
        <h2>Bonjour {prenom} {nom},</h2>
        <p>Voici votre code de vérification à usage unique :</p>
        <h1 style="font-size:48px;letter-spacing:12px;color:#4CAF50;
                   background:#f4f4f4;padding:20px 30px;border-radius:8px;
                   display:inline-block;">{otp_code}</h1>
        <p>⏱️ Ce code est valable <strong>10 minutes</strong>.</p>
        <p style="color:#999;font-size:12px;">Si vous n'avez pas demandé ce code, ignorez cet email.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())


# =========================
# JWT HELPERS
# =========================
def create_jwt(nom: str, prenom: str, email: str) -> str:
    payload = {
        "nom":    nom,
        "prenom": prenom,
        "email":  email,
        "iat":    datetime.datetime.utcnow(),
        "exp":    datetime.datetime.utcnow() + datetime.timedelta(hours=24),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="💰",
    layout="wide"
)

# =========================
# USER FORM  →  OTP  →  JWT
# =========================
def check_user() -> bool:
    """Multi-step authentication: info → OTP email → JWT session."""

    # ── Init session keys ──────────────────────────────────────────
    for key, default in [
        ("user_valid", False),
        ("otp_sent",   False),
        ("jwt_token",  None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Already logged in via JWT? ──────────────────────────────────
    if not st.session_state.user_valid and st.session_state.jwt_token:
        payload = verify_jwt(st.session_state.jwt_token)
        if payload:
            st.session_state.user_valid = True
            st.session_state.user_info  = {
                "nom":    payload["nom"],
                "prenom": payload["prenom"],
                "email":  payload["email"],
            }

    if st.session_state.user_valid:
        return True

    # ── Auth UI ────────────────────────────────────────────────────
    st.markdown("## 🔐 Authentification")
    st.markdown("---")

    # ── STEP 1 : collect user info & send OTP ──────────────────────
    if not st.session_state.otp_sent:
        with st.form("user_form"):
            nom    = st.text_input("Nom")
            prenom = st.text_input("Prénom")
            email  = st.text_input("Email")

            submitted = st.form_submit_button("📧 Envoyer le code de vérification",
                                              type="primary", use_container_width=True)

            if submitted:
                if not (nom and prenom and email):
                    st.error("⚠️ Veuillez remplir tous les champs.")
                else:
                    otp = str(random.randint(100000, 999999))
                    st.session_state["otp_code"]    = otp
                    st.session_state["otp_expiry"]  = (
                        datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                    )
                    st.session_state["pending_user"] = {
                        "nom": nom, "prenom": prenom, "email": email
                    }
                    with st.spinner("Envoi du code en cours…"):
                        try:
                            send_otp_email(email, otp, nom, prenom)
                            st.session_state.otp_sent = True
                            st.rerun()
                        except Exception as exc:
                            st.error(f"❌ Impossible d'envoyer l'email : {exc}")
                            st.caption(
                                "Vérifiez SMTP_EMAIL / SMTP_PASSWORD dans le fichier .env "
                                "et assurez-vous d'utiliser un *mot de passe d'application* Gmail."
                            )

    # ── STEP 2 : verify OTP ────────────────────────────────────────
    else:
        pending = st.session_state["pending_user"]
        st.success(f"📧 Un code à 6 chiffres a été envoyé à *{pending['email']}*")
        st.caption("Le code expire dans 10 minutes.")

        with st.form("otp_form"):
            otp_input = st.text_input("Code de vérification", max_chars=6,
                                      placeholder="123456")
            col_verify, col_resend = st.columns(2)
            verify  = col_verify.form_submit_button("✅ Valider",
                                                    type="primary",
                                                    use_container_width=True)
            resend  = col_resend.form_submit_button("🔄 Recommencer",
                                                    use_container_width=True)

            if resend:
                for k in ("otp_code", "otp_expiry", "otp_sent", "pending_user"):
                    st.session_state.pop(k, None)
                st.rerun()

            if verify:
                now = datetime.datetime.utcnow()
                if now > st.session_state["otp_expiry"]:
                    st.error("⏰ Code expiré. Veuillez recommencer.")
                    for k in ("otp_code", "otp_expiry", "otp_sent", "pending_user"):
                        st.session_state.pop(k, None)
                    st.rerun()
                elif otp_input == st.session_state["otp_code"]:
                    token = create_jwt(pending["nom"], pending["prenom"], pending["email"])
                    st.session_state.jwt_token  = token
                    st.session_state.user_valid = True
                    st.session_state.user_info  = pending
                    for k in ("otp_code", "otp_expiry", "otp_sent", "pending_user"):
                        st.session_state.pop(k, None)
                    st.rerun()
                else:
                    st.error("❌ Code incorrect. Réessayez.")

    return False

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

        st.info(f"Nombre total d'étapes : {len(all_steps)}")

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
