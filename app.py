import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="IPL Model Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏏 IPL Model Predictor")
st.markdown("---")

@st.cache_resource
def load_model(model_path):
    """
    Load the pickle model file
    
    Args:
        model_path: Path to the pickle file
        
    Returns:
        Loaded model object
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Sidebar for model loading
    st.sidebar.header("📁 Model Configuration")
    
    # File uploader for pickle model
    uploaded_file = st.sidebar.file_uploader(
        "Upload your pickle model file",
        type=['pkl', 'pickle'],
        help="Upload a trained model in pickle format"
    )
    
    # Alternative: Load from local path
    st.sidebar.markdown("### Or use local model")
    model_path = st.sidebar.text_input(
        "Enter model path",
        placeholder="e.g., models/model.pkl",
        help="Path to your local pickle file"
    )
    
    model = None
    
    # Load model from uploaded file
    if uploaded_file is not None:
        try:
            model = pickle.load(uploaded_file)
            st.sidebar.success("✅ Model loaded successfully from upload!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded model: {str(e)}")
    
    # Load model from local path
    elif model_path:
        if Path(model_path).exists():
            model = load_model(model_path)
            if model:
                st.sidebar.success("✅ Model loaded successfully from path!")
        else:
            st.sidebar.warning("⚠️ Model file not found at specified path")
    
    # Main content area
    if model is not None:
        st.success("🎉 Model is ready for predictions!")
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Information")
            st.write(f"**Model Type:** {type(model).__name__}")
            
            # Try to display model attributes
            if hasattr(model, 'get_params'):
                with st.expander("View Model Parameters"):
                    st.json(model.get_params())
        
        with col2:
            st.subheader("🔮 Make Predictions")
            st.info("Configure your input features below")
        
        # Prediction interface
        st.markdown("---")
        st.subheader("Input Features")
        
        # Example input fields - customize based on your model
        st.markdown("*Customize these inputs based on your model's requirements*")
        
        # Create sample input form
        with st.form("prediction_form"):
            num_features = st.number_input(
                "Number of features",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of input features your model expects"
            )
            
            # Dynamic feature inputs
            features = []
            cols = st.columns(min(num_features, 3))
            
            for i in range(num_features):
                with cols[i % 3]:
                    feature_val = st.number_input(
                        f"Feature {i+1}",
                        value=0.0,
                        key=f"feature_{i}"
                    )
                    features.append(feature_val)
            
            submit_button = st.form_submit_button("🚀 Predict")
            
            if submit_button:
                try:
                    # Create input array
                    input_data = np.array(features).reshape(1, -1)
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    
                    # Display prediction
                    st.success("### 🎯 Prediction Result")
                    st.write(prediction)
                    
                    # If probability prediction is available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_data)
                        st.write("**Prediction Probabilities:**")
                        st.write(proba)
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 Make sure your input features match the model's expected format")
    
    else:
        # Instructions when no model is loaded
        st.info("### 👋 Welcome! Get started by loading your model")
        
        st.markdown("""
        #### How to use this app:
        
        1. **Upload a model** using the sidebar file uploader, OR
        2. **Specify a local path** to your pickle model file
        3. Once loaded, configure your input features
        4. Click **Predict** to get results!
        
        #### Supported formats:
        - `.pkl` files
        - `.pickle` files
        
        #### Example model path:
        ```
        models/ipl_model.pkl
        ```
        """)
        
        # Create a models directory reminder
        st.warning("💡 **Tip:** Create a `models/` folder in your project to organize your model files!")

if __name__ == "__main__":
    main()
