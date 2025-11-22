import os
os.environ["STREAMLIT_WATCH_SYSTEM"] = "false"
os.environ["ST_DISABLE_FILE_WATCHER"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Academic Program Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .confidence-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .feature-card {
        padding: 15px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #1e3a8a;
    }
    h2 {
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Job classification function
job_groups = {
    'ai_data_ml': [
        'data scientist', 'associate data scientist', 'business analyst',
        'data analyst', 'data analytics', 'machine learning',
        'ml engineer', 'ai', 'cloud transformation'
    ],
    'cyber_security': [
        'security analyst', 'security engineer', 'cyber security',
        'application security', 'information security',
        'threat intelligence', 'network security'
    ],
    'software_engineering': [
        'software engineer', 'software design engineer',
        'system development engineer', 'developer', 'python',
        'full stack', 'qa engineer', 'engineer', 'digital engineer',
        'digital media analyst', 'geospatial analyst'
    ],
    'business_management': [
        'manager', 'management trainee', 'consultant-functional',
        'operations associate', 'academic associate', 'teacher',
        'ecologist', 'analyst', 'marketing', 'business development', 
        'sales trainee', 'sales development', 'business development trainee'
    ],
    'others': []
}

def classify_job(title):
    """Classify job title into job groups"""
    title = str(title).lower().strip()
    for group, keywords in job_groups.items():
        for keyword in keywords:
            if keyword.lower() in title:
                return group
    return 'others'

@st.cache_resource
def load_model():
    """Load the trained model and artifacts"""
    try:
        with open('program_prediction_model.pkl', 'rb') as f:
            model_artifacts = pickle.load(f)
        return model_artifacts
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please train the model first.")
        return None

def predict_program(company, job_role, package, model_artifacts):
    """Make prediction using the trained model"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    encoders = model_artifacts['encoders']
    
    features = {}
    
    # Handle Company
    if company:
        try:
            company_encoded = encoders['company'].transform([company])[0]
            features['company_encoded'] = company_encoded
        except ValueError:
            features['company_encoded'] = len(encoders['company'].classes_) // 2
    else:
        features['company_encoded'] = len(encoders['company'].classes_) // 2
    
    # Handle Job Role
    if job_role:
        job_group = classify_job(job_role)
        try:
            job_encoded = encoders['job'].transform([job_group])[0]
            features['job_group_encoded'] = job_encoded
        except ValueError:
            features['job_group_encoded'] = 0
    else:
        features['job_group_encoded'] = 0
    
    # Handle Package
    if package:
        package_group = pd.cut([package], bins=encoders['bins'], 
                              labels=encoders['labels'], include_lowest=True)[0]
        features['package_group'] = int(package_group)
    else:
        features['package_group'] = 2
    
    # Create and scale feature array
    feature_array = np.array([[
        features['company_encoded'],
        features['job_group_encoded'],
        features['package_group']
    ]])
    feature_scaled = scaler.transform(feature_array)
    
    # Make prediction
    prediction_encoded = model.predict(feature_scaled)[0]
    prediction_proba = model.predict_proba(feature_scaled)[0]
    prediction = encoders['program'].inverse_transform([prediction_encoded])[0]
    
    return {
        'predicted_program': prediction,
        'confidence': prediction_proba[prediction_encoded],
        'all_probabilities': {
            encoders['program'].classes_[i]: prob 
            for i, prob in enumerate(prediction_proba)
        },
        'job_group': classify_job(job_role) if job_role else 'Not provided'
    }

def create_probability_chart(probabilities):
    """Create an interactive bar chart for probabilities"""
    programs = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#667eea' if p == max(probs) else '#a5b4fc' for p in probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=programs,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Academic Program",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24}},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>🎓 Academic Program Predictor</h1>
            <p style='color: white; font-size: 18px; margin-top: 10px;'>
                AI-Powered Career Path Recommendation System
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=100)
        st.title("🔍 Input Features")
        st.markdown("---")
        
        # Model information
        with st.expander("ℹ️ About the Model", expanded=False):
            st.write(f"**Model Type:** {model_artifacts['model_name']}")
            st.write(f"**Features:** 3 (Company, Job Role, Package)")
            st.write(f"**Programs:** {len(model_artifacts['encoders']['program'].classes_)}")
            st.write("**Training:** GridSearchCV with 5-fold CV")
        
        st.markdown("---")
        
        # Input mode selection
        input_mode = st.radio(
            "Select Input Mode:",
            ["🎯 Guided Input", "✏️ Manual Entry"],
            help="Choose how you want to provide information"
        )
        
        st.markdown("---")
    
    # Main content area
    if input_mode == "🎯 Guided Input":
        st.subheader("📝 Enter Your Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏢 Company Information")
            companies = [''] + list(model_artifacts['encoders']['company'].classes_)
            company = st.selectbox(
                "Select Company",
                companies,
                help="Choose from the list of known companies"
            )
            
            if not company:
                st.info("💡 Tip: Selecting a company improves prediction accuracy!")
        
        with col2:
            st.markdown("### 💼 Job Details")
            job_role_examples = [
                '', 
                'Data Scientist', 
                'Software Engineer', 
                'Cyber Security Analyst',
                'Machine Learning Engineer',
                'Full Stack Developer',
                'Business Analyst',
                'Security Engineer'
            ]
            job_role = st.selectbox(
                "Select or Type Job Role",
                job_role_examples,
                help="Select from examples or type your own"
            )
            
            if not job_role:
                job_role_custom = st.text_input("Or enter custom job role:")
                if job_role_custom:
                    job_role = job_role_custom
        
        st.markdown("### 💰 Package Information")
        col3, col4 = st.columns([2, 1])
        
        with col3:
            package = st.slider(
                "Salary Package (LPA)",
                min_value=0.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Slide to select your package range"
            )
        
        with col4:
            st.metric("Selected Package", f"₹{package} LPA")
    
    else:  # Manual Entry
        st.subheader("✏️ Manual Entry Mode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            company = st.text_input(
                "Company Name",
                placeholder="e.g., IBM, Google, Microsoft",
                help="Enter company name (optional)"
            )
        
        with col2:
            job_role = st.text_input(
                "Job Role",
                placeholder="e.g., Data Scientist",
                help="Enter your job role (optional)"
            )
        
        with col3:
            package_input = st.text_input(
                "Package (LPA)",
                placeholder="e.g., 8.5",
                help="Enter package in lakhs (optional)"
            )
            
            package = float(package_input) if package_input else None
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🚀 Predict Program", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        if not company and not job_role and not package:
            st.warning("⚠️ Please provide at least one input feature for better predictions!")
        else:
            with st.spinner("🔮 Analyzing your profile..."):
                result = predict_program(
                    company if company else None,
                    job_role if job_role else None,
                    package,
                    model_artifacts
                )
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Main prediction result
            st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='color: white; margin: 0;'>Recommended Program</h2>
                    <h1 style='color: white; font-size: 48px; margin: 10px 0;'>
                        {result['predicted_program']}
                    </h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.1%}",
                    delta=f"{result['confidence']-0.7:.1%}" if result['confidence'] > 0.7 else None
                )
            
            with col2:
                st.metric(
                    "Company",
                    company if company else "Default",
                    delta="Provided" if company else "Default"
                )
            
            with col3:
                st.metric(
                    "Job Group",
                    result['job_group'].replace('_', ' ').title(),
                    delta="Classified" if job_role else "Default"
                )
            
            with col4:
                st.metric(
                    "Package",
                    f"₹{package if package else 'N/A'} LPA",
                    delta="Provided" if package else "Default"
                )
            
            st.markdown("---")
            
            # Detailed results
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("### 📊 Probability Distribution")
                fig_probs = create_probability_chart(result['all_probabilities'])
                st.plotly_chart(fig_probs, use_container_width=True)
            
            with col_right:
                st.markdown("### 🎯 Confidence Gauge")
                fig_gauge = create_confidence_gauge(result['confidence'])
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Detailed breakdown
            st.markdown("### 📋 Detailed Analysis")
            
            # Sort probabilities
            sorted_probs = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (program, prob) in enumerate(sorted_probs):
                if i == 0:
                    color = "#d1fae5"
                    icon = "🥇"
                elif i == 1:
                    color = "#fef3c7"
                    icon = "🥈"
                else:
                    color = "#fee2e2"
                    icon = "🥉"
                
                st.markdown(f"""
                    <div style='background-color: {color}; padding: 15px; border-radius: 8px; 
                    margin: 10px 0; border-left: 4px solid {"#4caf50" if i == 0 else "#ff9800" if i == 1 else "#f44336"}'>
                        <h4 style='margin: 0;'>{icon} {program}</h4>
                        <div style='background-color: #f0f0f0; border-radius: 10px; height: 30px; 
                        margin-top: 10px; overflow: hidden;'>
                            <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            height: 100%; width: {prob*100}%; display: flex; align-items: center; 
                            justify-content: center; color: white; font-weight: bold;'>
                                {prob:.1%}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if result['confidence'] > 0.8:
                st.success(f"""
                    ✅ **High Confidence Prediction!** 
                    
                    The model is very confident that **{result['predicted_program']}** is the right 
                    program for your profile. This recommendation is based on strong alignment 
                    between your inputs and historical data.
                """)
            elif result['confidence'] > 0.6:
                st.info(f"""
                    ℹ️ **Moderate Confidence Prediction**
                    
                    The model suggests **{result['predicted_program']}** with moderate confidence. 
                    Consider exploring the second-ranked option as well: 
                    **{sorted_probs[1][0]}** ({sorted_probs[1][1]:.1%})
                """)
            else:
                st.warning(f"""
                    ⚠️ **Lower Confidence Prediction**
                    
                    The prediction shows lower confidence. This might be due to:
                    - Limited input information provided
                    - Uncommon combination of features
                    - Consider providing more details for better accuracy
                    
                    Top 2 recommendations:
                    1. **{sorted_probs[0][0]}** ({sorted_probs[0][1]:.1%})
                    2. **{sorted_probs[1][0]}** ({sorted_probs[1][1]:.1%})
                """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 20px;'>
            <p>🎓 Academic Program Predictor | Powered by Machine Learning</p>
            <p style='font-size: 12px;'>Built with Streamlit, scikit-learn, and Plotly</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

