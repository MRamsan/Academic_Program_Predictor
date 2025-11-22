"""
Academic Program Predictor - Streamlit Application
Clean, production-ready code
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Academic Program Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .prob-bar {
        background-color: #e5e7eb;
        border-radius: 10px;
        height: 40px;
        margin-top: 10px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# JOB CLASSIFICATION
# ============================================================================
JOB_GROUPS = {
    'software_engineering': [
        'software engineer', 'software design engineer', 'system development engineer',
        'full stack developer', 'python developer', 'qa engineer', 'digital engineer'
    ],
    'data_science': [
        'data scientist', 'associate data scientist', 'data analyst', 'data analytics',
        'ai', 'ai-validation', 'ml engineer', 'machine learning', 'cloud transformation'
    ],
    'cyber_security': [
        'associate security engineer', 'security analyst', 'application security engineer',
        'cyber security engineer', 'cyber security', 'network security engineering',
        'information security', 'intelligence analyst', 'cyber threat intelligence'
    ],
    'marketing_sales': [
        'marketing', 'marketing executive', 'sales development', 'sales trainee',
        'business development', 'business development trainee'
    ],
    'business_management': [
        'manager', 'management trainee', 'consultant-functional', 'business analyst',
        'operations associate', 'academic associate', 'teacher', 'ecologist'
    ],
    'internship': ['internship with placement'],
    'others': ['geospatial analyst', 'digital media analyst', 'engineer']
}

def classify_job(title):
    """Classify job title into predefined groups"""
    if not title:
        return 'others'
    
    title = str(title).lower().strip()
    for group, keywords in JOB_GROUPS.items():
        for keyword in keywords:
            if keyword in title:
                return group
    return 'others'

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing artifacts"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return {'model': model, 'encoders': encoders, 'scaler': scaler}
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure model.pkl, encoders.pkl, and scaler.pkl are in the app directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_prediction(company, job_role, package, model_artifacts):
    """Make program prediction based on inputs"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    encoders = model_artifacts['encoders']
    
    # Process company
    company_encoded = 0
    if company:
        company = company.lower().strip()
        try:
            company_encoded = encoders['company'].transform([company])[0]
        except ValueError:
            # Unknown company - use median
            company_encoded = len(encoders['company'].classes_) // 2
    
    # Process job role
    job_group = classify_job(job_role)
    job_group_encoded = 0
    try:
        job_group_encoded = encoders['job_group'].transform([job_group])[0]
    except ValueError:
        job_group_encoded = 0
    
    # Process package
    package_group = 1  # Default to middle range
    if package is not None:
        if package <= 5:
            package_group = 0
        elif package <= 10:
            package_group = 1
        else:
            package_group = 2
    
    # Create feature array
    features = np.array([[company_encoded, job_group_encoded, package_group]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction_encoded = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Decode prediction
    predicted_program = encoders['program'].inverse_transform([prediction_encoded])[0]
    
    # Create probability dictionary
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        program = encoders['program'].inverse_transform([i])[0]
        prob_dict[program] = prob
    
    return {
        'predicted_program': predicted_program,
        'confidence': probabilities[prediction_encoded],
        'probabilities': prob_dict,
        'job_group': job_group
    }

# ============================================================================
# UI COMPONENTS
# ============================================================================
def display_header():
    """Display application header"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🎓 Academic Program Predictor</h1>
            <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>
                AI-Powered Career Path Recommendation System
            </p>
        </div>
    """, unsafe_allow_html=True)

def display_sidebar(model_artifacts):
    """Display sidebar with information"""
    with st.sidebar:
        st.markdown("### 🔍 About")
        st.info("""
            This AI-powered system predicts the most suitable academic program 
            based on your company, job role, and package information.
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.write(f"**Algorithm:** XGBoost Classifier")
        st.write(f"**Programs:** {len(model_artifacts['encoders']['program'].classes_)}")
        st.write(f"**Features:** 3 (Company, Job, Package)")
        
        st.markdown("---")
        
        st.markdown("### 📚 Available Programs")
        programs = sorted(model_artifacts['encoders']['program'].classes_)
        for prog in programs:
            st.write(f"• {prog.upper()}")

def display_results(result):
    """Display prediction results"""
    # Main prediction card
    st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Recommended Program</h2>
            <h1 style='color: white; font-size: 3rem; margin: 1rem 0;'>
                {result['predicted_program'].upper()}
            </h1>
            <p style='color: white; font-size: 1.2rem; margin: 0;'>
                Confidence: {result['confidence']:.1%}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "#22c55e" if result['confidence'] > 0.7 else "#f59e0b" if result['confidence'] > 0.5 else "#ef4444"
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {confidence_color}; margin: 0;'>{result['confidence']:.1%}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>Confidence Level</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #3b82f6; margin: 0;'>{result['job_group'].replace('_', ' ').title()}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>Job Category</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rank_2 = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[1]
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #8b5cf6; margin: 0;'>{rank_2[0].upper()}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>2nd Choice ({rank_2[1]:.1%})</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Probability breakdown
    st.markdown("---")
    st.markdown("### 📊 All Program Probabilities")
    
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (program, prob) in enumerate(sorted_probs):
        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📌"
        
        col_prog, col_bar = st.columns([1, 3])
        
        with col_prog:
            st.markdown(f"**{icon} {program.upper()}**")
        
        with col_bar:
            st.markdown(f"""
                <div class='prob-bar'>
                    <div class='prob-fill' style='width: {prob*100}%;'>
                        {prob:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Interpretation")
    
    if result['confidence'] > 0.75:
        st.success(f"""
            **High Confidence Prediction** ✅
            
            The model is highly confident that **{result['predicted_program'].upper()}** is the best 
            fit for your profile. This recommendation is based on strong patterns in the training data.
        """)
    elif result['confidence'] > 0.55:
        st.info(f"""
            **Moderate Confidence Prediction** ℹ️
            
            The model suggests **{result['predicted_program'].upper()}** as the top choice, but also 
            recommends considering **{sorted_probs[1][0].upper()}** ({sorted_probs[1][1]:.1%}) as an 
            alternative option.
        """)
    else:
        st.warning(f"""
            **Lower Confidence Prediction** ⚠️
            
            The prediction shows moderate confidence. Consider:
            - Providing more specific information
            - Exploring multiple program options
            
            Top recommendations:
            1. **{sorted_probs[0][0].upper()}** ({sorted_probs[0][1]:.1%})
            2. **{sorted_probs[1][0].upper()}** ({sorted_probs[1][1]:.1%})
        """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    # Display header
    display_header()
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts is None:
        st.stop()
    
    # Display sidebar
    display_sidebar(model_artifacts)
    
    # Main input section
    st.markdown("## 📝 Enter Your Information")
    st.markdown("Provide at least one of the following fields for better predictions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Company")
        companies = [''] + sorted(list(model_artifacts['encoders']['company'].classes_))
        company = st.selectbox(
            "Select or type company name",
            companies,
            help="Choose from known companies or leave blank"
        )
        
        if not company:
            company_custom = st.text_input("Or enter company name:", placeholder="e.g., IBM, Google")
            if company_custom:
                company = company_custom
        
        st.markdown("#### 💼 Job Role")
        job_examples = [
            '',
            'Data Scientist',
            'Software Engineer',
            'Cyber Security Analyst',
            'Business Analyst',
            'Full Stack Developer',
            'ML Engineer'
        ]
        job_role = st.selectbox(
            "Select or type job role",
            job_examples,
            help="Select from examples or enter custom role"
        )
        
        if not job_role:
            job_custom = st.text_input("Or enter job role:", placeholder="e.g., Python Developer")
            if job_custom:
                job_role = job_custom
    
    with col2:
        st.markdown("#### 💰 Package Information")
        
        package_input_type = st.radio(
            "Input method:",
            ["Slider", "Text Input"],
            horizontal=True
        )
        
        if package_input_type == "Slider":
            package = st.slider(
                "Annual Package (LPA)",
                min_value=0.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Select your salary package"
            )
        else:
            package_text = st.text_input(
                "Enter package (LPA):",
                placeholder="e.g., 8.5",
                help="Enter package in lakhs per annum"
            )
            package = float(package_text) if package_text else None
        
        if package:
            st.metric("Selected Package", f"₹{package} LPA")
            
            # Show package range
            if package <= 5:
                st.info("📊 Range: 0-5 LPA")
            elif package <= 10:
                st.info("📊 Range: 5-10 LPA")
            else:
                st.info("📊 Range: 10+ LPA")
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🚀 Predict Program", use_container_width=True)
    
    # Make prediction
    if predict_button:
        if not company and not job_role and not package:
            st.warning("⚠️ Please provide at least one input for prediction!")
        else:
            with st.spinner("🔮 Analyzing your profile..."):
                result = make_prediction(company, job_role, package, model_artifacts)
            
            st.success("✅ Prediction Complete!")
            display_results(result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 1rem;'>
            <p>🎓 Academic Program Predictor | Powered by Machine Learning</p>
            <p style='font-size: 0.875rem;'>Built with Streamlit & XGBoost</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
