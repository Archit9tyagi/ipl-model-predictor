# IPL Model Predictor 🏏

A Streamlit web application for loading and running IPL prediction models via pickle files.

## Features

- 📁 Load models from pickle files (.pkl, .pickle)
- 🔼 Upload models directly through the web interface
- 🔮 Make predictions with custom input features
- 📊 View model information and parameters
- 🎨 Clean and intuitive user interface

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

### Loading a Model

You have two options to load your pickle model:

1. **Upload via interface**: Use the file uploader in the sidebar
2. **Local path**: Enter the path to your model file (e.g., `models/model.pkl`)

### Making Predictions

1. Once your model is loaded, configure the input features
2. Enter values for each feature your model expects
3. Click the "Predict" button to get results

## Project Structure

```
Final_IPL/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
└── models/            # Directory for your model files (create as needed)
```

## Dependencies

- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `pickle5` - Pickle protocol support

## Contributing

Feel free to customize the app.py file to match your specific model requirements and add additional features as needed.

## License

MIT License
