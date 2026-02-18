import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import hashlib

# Configure the page
st.set_page_config(
    page_title="Startup Profit Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Authentication
def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256("admin123".encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("### 🔐 Authentication Required")
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password",
                help="Default password: admin123"
            )
            st.info("💡 Default credentials - Password: *admin123*")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.markdown("### 🔐 Authentication Required")
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password"
            )
            st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Load and prepare data
@st.cache_data
def load_data():
    """Load the startup data"""
    try:
        df = pd.read_csv('50_Startups.csv')
        return df
    except FileNotFoundError:
        # If file not found, create sample data
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
    """Train the model using backward elimination as in the notebook"""
    # Prepare data
    data = df.copy()
    
    # Encode State
    le = LabelEncoder()
    data['State'] = le.fit_transform(data['State'])
    
    # Prepare features
    X = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']].values
    y = data['Profit'].values
    
    # Add constant and perform backward elimination
    X_with_const = sm.add_constant(X)
    
    # Based on your notebook, the final model uses only const and R&D Spend (index 1)
    X_opt = X_with_const[:, [0, 1]]  # const and R&D Spend
    
    # Fit the final model
    model = sm.OLS(endog=y, exog=X_opt).fit()
    
    return model, le

def validate_inputs(rd_spend, admin, marketing, state_input):
    """Validate user inputs"""
    errors = []
    
    # Validate R&D Spend
    if rd_spend < 0:
        errors.append("❌ R&D Spend cannot be negative")
    if rd_spend > 200000:
        errors.append("⚠️ R&D Spend seems unusually high (max in dataset: $165,349)")
    
    # Validate Administration
    if admin < 0:
        errors.append("❌ Administration cannot be negative")
    if admin > 200000:
        errors.append("⚠️ Administration seems unusually high (max in dataset: $182,646)")
    
    # Validate Marketing Spend
    if marketing < 0:
        errors.append("❌ Marketing Spend cannot be negative")
    if marketing > 500000:
        errors.append("⚠️ Marketing Spend seems unusually high (max in dataset: $471,784)")
    
    # Validate State
    valid_states = ['C', 'N', 'F']
    if state_input.upper() not in valid_states:
        errors.append(f"❌ State must be one of: C (California), N (New York), F (Florida)")
    
    return errors

def state_code_to_name(code):
    """Convert state code to full name"""
    mapping = {
        'C': 'California',
        'N': 'New York',
        'F': 'Florida'
    }
    return mapping.get(code.upper(), code)

# Main application
if check_password():
    # Load data and model
    try:
        df = load_data()
        model, label_encoder = train_model(df)
        
        # Logout button in sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/profit.png", width=80)
            st.markdown("### 💰 Profit Predictor")
            st.markdown("---")
            st.markdown("*State Codes:*")
            st.markdown("- *C* = California 🌴")
            st.markdown("- *N* = New York 🗽")
            st.markdown("- *F* = Florida 🏖️")
            st.markdown("---")
            st.header("📊 Model Stats")
            st.metric("R²", f"{model.rsquared:.4f}")
            st.metric("Adj. R²", f"{model.rsquared_adj:.4f}")
            st.metric("Observations", len(df))
            
            if st.checkbox("Show Model Summary"):
                st.text(model.summary().as_text())
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state["password_correct"] = False
                st.rerun()
        
        # Main interface
        st.title("🚀 Startup Profit Prediction System")
        st.markdown("### Predict startup profit based on spending and location")
        st.markdown("---")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Input Variables")
            
            # Input fields
            rd_spend = st.number_input(
                "💡 R&D Spend ($)",
                min_value=0.0,
                max_value=500000.0,
                value=100000.0,
                step=1000.0,
                help="Research and Development spending"
            )
            
            administration = st.number_input(
                "📋 Administration ($)",
                min_value=0.0,
                max_value=500000.0,
                value=120000.0,
                step=1000.0,
                help="Administrative spending"
            )
            
            marketing_spend = st.number_input(
                "📢 Marketing Spend ($)",
                min_value=0.0,
                max_value=500000.0,
                value=200000.0,
                step=1000.0,
                help="Marketing spending"
            )
            
            state_input = st.text_input(
                "📍 State Code",
                value="C",
                max_chars=1,
                help="Enter: C (California), N (New York), or F (Florida)"
            ).upper()
            
            # Display state name
            if state_input in ['C', 'N', 'F']:
                st.success(f"✅ Selected State: *{state_code_to_name(state_input)}*")
        
        with col2:
            st.subheader("🎯 Predictions")
            
            # Validate inputs
            validation_errors = validate_inputs(rd_spend, administration, marketing_spend, state_input)
            
            if validation_errors:
                st.error("*Validation Errors:*")
                for error in validation_errors:
                    st.write(error)
            else:
                st.success("✅ All inputs are valid")
            
            # Prepare input for prediction
            new_input = np.array([[1.0, rd_spend]])  # [const, R&D Spend]
            
            # Create prediction history in session state
            if 'predictions' not in st.session_state:
                st.session_state.predictions = []
            
            # Buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📊 Show All", use_container_width=True, disabled=bool(validation_errors), type="primary"):
                    # Make prediction
                    prediction = model.predict(new_input)[0]
                    
                    # Store prediction
                    prediction_data = {
                        'R&D Spend': rd_spend,
                        'Administration': administration,
                        'Marketing Spend': marketing_spend,
                        'State': state_code_to_name(state_input),
                        'Predicted Profit': prediction
                    }
                    st.session_state.predictions.append(prediction_data)
            
            with col_btn2:
                if st.button("🎯 Last Only", use_container_width=True, disabled=bool(validation_errors), type="secondary"):
                    # Make prediction
                    prediction = model.predict(new_input)[0]
                    
                    # Store prediction
                    prediction_data = {
                        'R&D Spend': rd_spend,
                        'Administration': administration,
                        'Marketing Spend': marketing_spend,
                        'State': state_code_to_name(state_input),
                        'Predicted Profit': prediction
                    }
                    st.session_state.predictions.append(prediction_data)
        
        # Display predictions
        st.markdown("---")
        if st.session_state.predictions:
            if len(st.session_state.predictions) > 0:
                last_prediction = st.session_state.predictions[-1]
                
                st.subheader("📈 Latest Prediction Result")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("💰 Profit", f"${last_prediction['Predicted Profit']:,.2f}")
                with col_b:
                    st.metric("💡 R&D", f"${last_prediction['R&D Spend']:,.2f}")
                with col_c:
                    st.metric("📋 Admin", f"${last_prediction['Administration']:,.2f}")
                with col_d:
                    st.metric("📢 Marketing", f"${last_prediction['Marketing Spend']:,.2f}")
                
                st.info(f"📍 Location: *{last_prediction['State']}*")
                
                # Show all predictions table
                if len(st.session_state.predictions) > 1:
                    with st.expander(f"📋 View All {len(st.session_state.predictions)} Predictions"):
                        predictions_df = pd.DataFrame(st.session_state.predictions)
                        st.dataframe(predictions_df.style.format({
                            'R&D Spend': '${:,.2f}',
                            'Administration': '${:,.2f}',
                            'Marketing Spend': '${:,.2f}',
                            'Predicted Profit': '${:,.2f}'
                        }), use_container_width=True)
                        
                        # Download option
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name="startup_predictions.csv",
                            mime="text/csv"
                        )
                
                # Clear button
                if st.button("🗑️ Clear All Predictions"):
                    st.session_state.predictions = []
                    st.success("✅ Cleared!")
                    st.rerun()
        
        # Dataset overview
        st.markdown("---")
        st.subheader("📊 Dataset Overview")
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("🏢 Total Startups", len(df))
        
        with col4:
            st.metric("📊 Avg Profit", f"${df['Profit'].mean():,.0f}")
        
        with col5:
            st.metric("📉 Min Profit", f"${df['Profit'].min():,.0f}")
        
        with col6:
            st.metric("📈 Max Profit", f"${df['Profit'].max():,.0f}")
        
        # Dataset view
        with st.expander("🔍 View Complete Dataset"):
            st.dataframe(df.style.format({
                'R&D Spend': '${:,.2f}',
                'Administration': '${:,.2f}',
                'Marketing Spend': '${:,.2f}',
                'Profit': '${:,.2f}'
            }), use_container_width=True)
            
            # State distribution
            st.markdown("*State Distribution:*")
            state_counts = df['State'].value_counts()
            st.bar_chart(state_counts)
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit | © 2026 Startup Profit Predictor")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

