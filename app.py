import streamlit as st
import pickle
import numpy as np
import os
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).parent.absolute()

# Path to ml_models folder
models_dir = current_dir / "ml_models"

# 1. Load all 4 model assets with proper error handling
@st.cache_resource
def load_assets():
    # List of required files
    required_files = ['scaler.pkl', 'svm_model.pkl', 'se_estimator.pkl', 'worst_estimator.pkl']
    
    # Check if ml_models folder exists
    if not models_dir.exists():
        st.error(f"❌ 'ml_odels' folder not found at: {models_dir}")
        st.info("Please ensure you have a folder named 'ml_models' containing your model files.")
        return None, None, None, None
    
    # Check if all files exist in the ml_models folder
    missing_files = []
    for file in required_files:
        if not os.path.exists(models_dir / file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"❌ Missing required model files in 'ml_models' folder: {', '.join(missing_files)}")
        st.info(f"""
        Please ensure the following files are in the 'ml_models' folder:
        - scaler.pkl
        - svm_model.pkl  
        - se_estimator.pkl
        - worst_estimator.pkl
        
        **Troubleshooting steps:**
        1. Check if the files are in the 'ml_models' folder
        2. Verify the filenames match exactly (case-sensitive)
        3. Make sure you've trained and saved your models
        """)
        
        # Show what's actually in the ml_models folder
        st.write(f"**ml_models folder contents:**")
        files = os.listdir(models_dir)
        for f in files:
            st.write(f"- {f}")
        return None, None, None, None
    
    try:
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(models_dir / 'svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(models_dir / 'se_estimator.pkl', 'rb') as f:
            se_estimator = pickle.load(f)
        with open(models_dir / 'worst_estimator.pkl', 'rb') as f:
            worst_estimator = pickle.load(f)
        
        st.success("✅ All models loaded successfully!")
        return scaler, model, se_estimator, worst_estimator
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        return None, None, None, None

# Load assets
scaler, model, se_estimator, worst_estimator = load_assets()

# Check if models loaded successfully
if scaler is None:
    st.stop()  # Stop execution if models didn't load

# Page configurations
st.set_page_config(page_title="Breast Cancer Diagnostics", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Diagnostic System")
st.write("Select your data entry method below to evaluate tumor clinical characteristics.")

# Initialize session state
if 'full_30_features' not in st.session_state:
    st.session_state.full_30_features = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'previous_mode' not in st.session_state:
    st.session_state.previous_mode = None
if 'mode_switched' not in st.session_state:
    st.session_state.mode_switched = False

# 2. Mode Selector
entry_mode = st.sidebar.radio(
    "Choose Input Method:",
    ["Simulated Quick Mode (10 Core Sliders)", "Full Laboratory Mode (30 Exact Metrics)"],
    horizontal=True
)

# Check if mode changed - if yes, reset EVERYTHING
if st.session_state.previous_mode != entry_mode:
    # Reset all prediction-related state
    st.session_state.full_30_features = None
    st.session_state.prediction_made = False
    st.session_state.show_results = False
    st.session_state.mode_switched = True
    st.session_state.previous_mode = entry_mode
    # Force a complete rerun to clear all widgets
    st.rerun()
else:
    st.session_state.mode_switched = False

# Universal list of all 30 feature names in the exact dataset order
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# ==========================================
# OPTION A: 10 CORE SLIDERS WITH ML PREDICTION
# ==========================================
if entry_mode == "Simulated Quick Mode (10 Core Sliders)":
    st.subheader("⚡ Quick Mode: 10 Core Measurements")
    st.info("💡 Enter the 10 core mean measurements. The system will predict the error and worst values automatically.")
    
    # Use a form to prevent automatic reruns
    with st.form(key="quick_mode_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📐 Size & Shape Metrics")
            mean_radius = st.number_input("Radius (Mean size of cells)", value=14.0, step=0.1, format="%.7f", key="quick_radius")
            mean_perimeter = st.number_input("Perimeter (Mean core tumor perimeter)", value=90.0, step=0.5, format="%.7f", key="quick_perimeter")
            mean_area = st.number_input("Area (Mean core tumor area)", value=600.0, step=10.0, format="%.7f", key="quick_area")
            mean_smoothness = st.number_input("Smoothness (Variation in radius lengths)", value=0.1, step=0.01, format="%.7f", key="quick_smoothness")
            mean_compactness = st.number_input("Compactness (Perimeter² / Area - 1.0)", value=0.1, step=0.01, format="%.7f", key="quick_compactness")

        with col2:
            st.markdown("##### 🔬 Surface & Texture Metrics")
            mean_concavity = st.number_input("Concavity (Severity of concave contours)", value=0.1, step=0.01, format="%.7f", key="quick_concavity")
            mean_concave_points = st.number_input("Concave Points (Number of concave portions)", value=0.05, step=0.01, format="%.7f", key="quick_concave_points")
            mean_fractal_dimension = st.number_input("Fractal Dimension ('Coastline approximation')", value=0.06, step=0.01, format="%.7f", key="quick_fractal")
            mean_texture = st.number_input("Texture (Variance of gray-scale values)", value=20.0, step=0.5, format="%.7f", key="quick_texture")
            mean_symmetry = st.number_input("Symmetry Score", value=0.18, step=0.01, format="%.7f", key="quick_symmetry")

        # Quick mode button inside form
        submitted = st.form_submit_button("⚡ Quick Analyze (Predict from 10 inputs)", type="primary", use_container_width=True)
        
        if submitted:
            user_means = np.array([[
                mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension
            ]])
            
            predicted_se = se_estimator.predict(user_means)
            predicted_worst = worst_estimator.predict(user_means)
            st.session_state.full_30_features = np.hstack([user_means, predicted_se, predicted_worst])
            st.session_state.prediction_made = True
            st.session_state.show_results = True
            st.rerun()

# ==========================================
# OPTION B: 30 EXACT METRIC NUMERIC INPUTS
# ==========================================
else:
    st.subheader("🔬 Full Mode: 30 Exact Metrics")
    st.write("Enter the exact clinical data points recorded from the biopsy evaluation.")
    
    # Use a form to prevent automatic reruns
    with st.form(key="full_mode_form"):
        # Organize into 3 neat columns: Means, Standard Errors, and Worst values
        lab_col1, lab_col2, lab_col3 = st.columns(3)
        user_inputs = []
        
        for idx, name in enumerate(feature_names):
            display_label = name.replace(" ", " ").title()
            
            # Determine default values - using realistic values
            if idx < 10:  # Mean values
                if "radius" in name:
                    default_val = 14.0
                elif "texture" in name:
                    default_val = 20.0
                elif "perimeter" in name:
                    default_val = 90.0
                elif "area" in name:
                    default_val = 600.0
                elif "smoothness" in name:
                    default_val = 0.1
                elif "compactness" in name:
                    default_val = 0.1
                elif "concavity" in name:
                    default_val = 0.1
                elif "concave points" in name:
                    default_val = 0.05
                elif "symmetry" in name:
                    default_val = 0.18
                else:  # fractal dimension
                    default_val = 0.06
            else:  # Error and Worst values - smaller defaults
                default_val = 0.01
            
            # Route feature inputs into their corresponding column layout
            if idx < 10:
                with lab_col1:
                    val = st.number_input(display_label, value=default_val, format="%.4f", key=f"lab_full_{idx}")
            elif idx < 20:
                with lab_col2:
                    val = st.number_input(display_label, value=default_val, format="%.4f", key=f"lab_full_{idx}")
            else:
                with lab_col3:
                    val = st.number_input(display_label, value=default_val, format="%.4f", key=f"lab_full_{idx}")
                    
            user_inputs.append(val)
        
        # Full mode button inside form
        submitted = st.form_submit_button("🔬 Full Analyze (Use all 30 inputs)", type="primary", use_container_width=True)
        
        if submitted:
            st.session_state.full_30_features = np.array(user_inputs).reshape(1, -1)
            st.session_state.prediction_made = True
            st.session_state.show_results = True
            st.rerun()

# ==========================================
# COMMON PREDICTION ENGINE - DISPLAY RESULTS
# ==========================================
# Only show results if we're NOT in a mode switch and results exist
if not st.session_state.mode_switched and st.session_state.show_results and st.session_state.full_30_features is not None:
    try:
        # Process the 30-feature vector
        scaled_features = scaler.transform(st.session_state.full_30_features)
        prediction = model.predict(scaled_features)
        
        # Get prediction probability if available
        try:
            proba = model.predict_proba(scaled_features)
            confidence = proba[0][1] if prediction[0] == 1 else proba[0][0]
        except:
            confidence = None
        
        st.markdown("---")
        st.subheader("📊 Diagnostic Verdict")
        
        # Display results with more detail
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ **Malignant (Indicative of Malignancy)**")
                st.write("""The clinical properties align significantly with patterns observed in cancerous cell samples.
                          Immediate medical validation recommended.""")
            else:
                st.success("✅ **Benign (Indicative of Healthy Tissue)**")
                st.write("The structural parameters track closely within standard, non-cancerous baseline properties.")
        
        with col2:
            if confidence is not None:
                st.metric("Confidence Score", f"{confidence*100:.1f}%")
                # Add a visual indicator
                if confidence > 0.9:
                    st.info("📊 High confidence prediction")
                elif confidence > 0.7:
                    st.info("📊 Moderate confidence prediction")
                else:
                    st.warning("📊 Low confidence - consider additional tests")
        
        # Show feature summary (optional)
        with st.expander("📋 View Full Feature Vector"):
            st.write("The 30 features used for prediction:")
            features_dict = {feature_names[i]: float(st.session_state.full_30_features[0][i]) for i in range(30)}
            # Format to show only 4 decimal places for readability
            formatted_dict = {k: f"{v:.4f}" for k, v in features_dict.items()}
            st.json(formatted_dict)
        
        # Reset button
        if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
            st.session_state.full_30_features = None
            st.session_state.prediction_made = False
            st.session_state.show_results = False
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please try again with different input values.")
        if st.button("🔄 Reset", type="secondary"):
            st.session_state.full_30_features = None
            st.session_state.prediction_made = False
            st.session_state.show_results = False
            st.rerun()

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("""⚠️ **Disclaimer:** This system is for educational and research purposes only. 
           Always consult with a qualified healthcare professional for medical diagnoses.""")