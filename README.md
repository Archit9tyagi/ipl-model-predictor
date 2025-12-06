# IPL Win Predictor 🏏

A beautiful Streamlit web application for predicting IPL match outcomes using machine learning.

## Features

- 🎨 **Premium Dark Theme UI** - Modern design with glassmorphism effects
- 🏏 **Cricket-Specific Validation** - Proper overs format (0.0 to 20.5)
- 📊 **Real-time Predictions** - Instant win probability calculations
- 📈 **Detailed Statistics** - Current run rate, required run rate, pressure index
- 📱 **Fully Responsive** - Works perfectly on mobile, tablet, and desktop
- ⚡ **Simple Setup** - No virtual environment required for basic usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

That's it! Only 4 packages needed:
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning

### 2. Run the Application

```bash
streamlit run app_streamlit.py
```

The app will automatically open in your browser at `http://localhost:8501`

## How to Use

### Making Predictions

1. **Select Teams**: Choose the batting and bowling teams from the dropdowns
2. **Select Venue**: Pick the host city for the match
3. **Enter Match State**:
   - Current Score (runs scored so far)
   - Wickets (wickets fallen)
   - Overs (format: 14.2 means 14 overs and 2 balls)
   - Target (runs to win)
4. **Click Predict**: Get instant win probability!

### Understanding the Output

- **Win Probability Circle**: Shows batting team's chance of winning
- **Comparison Bar**: Visual comparison of both teams' win chances
- **Statistics Cards**:
  - **Required Run Rate (RRR)**: Runs per over needed to win
  - **Current Run Rate (CRR)**: Current scoring rate
  - **Remaining Balls**: Balls left in the innings
  - **Pressure Index**: Low (green) / Medium (yellow) / High (red)

## Cricket-Specific Features

### Overs Validation
The app automatically validates cricket overs format:
- Valid: `10.0` to `10.5` (10 overs, 0-5 balls)
- Invalid: `10.6` or higher (auto-converts to next over)
- Maximum: `20.0` overs (T20 format)

### Match End Detection
The app will alert you if:
- 20 overs are completed
- No balls remaining
- Invalid target (≤ 0)

## Project Structure

```
Final_IPL/
├── app_streamlit.py      # Main Streamlit application
├── pipe.pkl              # Trained ML model (Logistic Regression)
├── deliveries.csv        # Ball-by-ball IPL data
├── matches.csv           # IPL match summaries
├── Untitled.ipynb        # Jupyter notebook for model training
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Supported Teams

- Sunrisers Hyderabad
- Mumbai Indians
- Royal Challengers Bangalore
- Kolkata Knight Riders
- Kings XI Punjab
- Chennai Super Kings
- Rajasthan Royals
- Delhi Capitals

## Supported Venues

29 venues including major Indian and international cities:
- Indian: Hyderabad, Bangalore, Mumbai, Kolkata, Delhi, Chennai, Pune, etc.
- International: Cape Town, Durban, Johannesburg, Abu Dhabi, Sharjah, etc.

## Technical Details

### Machine Learning Model
- **Algorithm**: Logistic Regression
- **Features**: Teams, venue, runs left, balls left, wickets, current run rate, required run rate
- **Output**: Win probability for both teams (0-100%)

### UI Technology
- **Framework**: Streamlit
- **Styling**: Custom CSS with dark theme
- **Font**: Lexend (Google Fonts)
- **Design**: Glassmorphism with smooth animations

## Optional: Using Virtual Environment

While not required, you can use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app_streamlit.py
```

## Data Sources

The prediction model is trained on historical IPL data:
- **deliveries.csv**: ~179,000 ball-by-ball records
- **matches.csv**: ~757 match summaries

## Troubleshooting

### Model Not Found Error
Make sure `pipe.pkl` is in the same directory as `app_streamlit.py`

### Import Errors
Run `pip install -r requirements.txt` to install all dependencies

### Port Already in Use
If port 8501 is busy, Streamlit will automatically try the next available port

## Contributing

Feel free to:
- Report bugs
- Suggest features
- Improve the ML model
- Enhance the UI design

## License

MIT License

---

**Made with ❤️ for Cricket Fans**

Powered by Machine Learning | Built with Streamlit
