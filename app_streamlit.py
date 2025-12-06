import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import time

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------
defaults = {
    "theme": "dark",
    "score": 124,
    "wickets": 3,
    "target": 189,
    "overs": 14,
    "balls": 2,
    "prediction_made": False,
    "show_loading": False,
    "toast_message": None,
    "toast_type": "error",
    "win": 0,
    "loss": 0,
    "batting_team": None,
    "bowling_team": None
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------
# STATIC DATA
# ---------------------------------------------------------
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

team_icons = {
    'Sunrisers Hyderabad': '🌅',
    'Mumbai Indians': '🔵',
    'Royal Challengers Bangalore': '❤️‍🔥',
    'Kolkata Knight Riders': '💜',
    'Kings XI Punjab': '❤️',
    'Chennai Super Kings': '💛',
    'Rajasthan Royals': '💙',
    'Delhi Capitals': '🔷'
}

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource(ttl=None)
def load_model():
    """Load the machine learning model with error handling"""
    try:
        model_path = "pipe.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file (pipe.pkl) not found!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        import traceback
        st.error(f"```\n{traceback.format_exc()}\n```")
        return None

pipe = load_model()

# ---------------------------------------------------------
# THEMES
# ---------------------------------------------------------
themes = {
    'dark': {
        'bg_gradient': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
        'text': 'white',
        'card': 'rgba(255,255,255,0.05)',
        'border': 'rgba(255,255,255,0.2)',
        'primary': '#3b82f6',
        'secondary': '#a855f7',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b'
    },
    'light': {
        'bg_gradient': 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f8fafc 100%)',
        'text': '#1e293b',
        'card': 'rgba(255,255,255,0.95)',
        'border': 'rgba(0,0,0,0.1)',
        'primary': '#3b82f6',
        'secondary': '#a855f7',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b'
    }
}

T = themes[st.session_state.theme]

# ---------------------------------------------------------
# CSS STYLES
# ---------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: {T['bg_gradient']};
        color: {T['text']};
    }}

    /* Hide Streamlit Branding */
    #MainMenu, footer {{visibility: hidden;}}
    
    /* Hide default padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* MAIN TITLE */
    .title {{
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {T['primary']} 0%, {T['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: slideDown 0.6s ease-out;
    }}
    
    /* Team VS Header */
    .vs-header {{
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0;
        padding: 1rem;
        background: {T['card']};
        border-radius: 16px;
        border: 1px solid {T['border']};
        backdrop-filter: blur(10px);
    }}
    
    .vs-text {{
        color: {T['secondary']};
        font-size: 1.2rem;
        margin: 0 1rem;
    }}

    /* CARDS */
    .card {{
        background: {T['card']};
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid {T['border']};
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.15);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: {T['card']};
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid {T['border']};
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 0.85rem;
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {T['primary']};
    }}

    /* INPUT LABELS */
    .input-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {T['text']};
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-align: center;
    }}

    /* TOAST NOTIFICATIONS */
    .toast {{
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 12px;
        font-weight: 600;
        color: white;
        z-index: 9999;
        animation: slideIn 0.3s ease-out, fadeOut 0.5s ease-out 2.5s forwards;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .toast-error {{ background: {T['danger']}; }}
    .toast-warning {{ background: {T['warning']}; }}
    .toast-success {{ background: {T['success']}; }}

    @keyframes slideIn {{
        from {{ transform: translateX(400px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    @keyframes fadeOut {{
        to {{ opacity: 0; transform: translateX(100px); }}
    }}
    
    @keyframes slideDown {{
        from {{ transform: translateY(-50px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* WIN PROBABILITY RING */
    .ring {{
        width: 280px;
        height: 280px;
        margin: 2rem auto;
        position: relative;
    }}
    .ring-center {{
        position: absolute;
        top: 50%; 
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }}
    
    .ring-percentage {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {T['primary']} 0%, {T['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .ring-label {{
        opacity: 0.7;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    /* Match Ended Card */
    .match-ended {{
        background: linear-gradient(135deg, {T['danger']} 0%, {T['warning']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        animation: pulse 2s infinite;
        margin: 2rem 0;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
    
    /* Probability Bars */
    .prob-bar {{
        height: 40px;
        border-radius: 20px;
        overflow: hidden;
        background: {T['card']};
        border: 1px solid {T['border']};
        margin: 0.5rem 0;
        position: relative;
    }}
    
    .prob-fill {{
        height: 100%;
        transition: width 1s ease-out;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 1rem;
        font-weight: 700;
        color: white;
    }}
    
    /* Theme Toggle */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        left: 20px;
        background: {T['card']};
        border: 1px solid {T['border']};
        padding: 8px 12px;
        border-radius: 12px;
        cursor: pointer;
        backdrop-filter: blur(10px);
        z-index: 999;
        font-size: 1.2rem;
    }}
    
    /* Stepper Controls */
    .stepper {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .stepper-btn {{
        background: {T['primary']};
        color: white;
        border: none;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-size: 1.5rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .stepper-btn:hover {{
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }}
    
    .stepper-value {{
        font-size: 2rem;
        font-weight: 700;
        min-width: 100px;
        text-align: center;
    }}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def toast(msg, type="error"):
    """Display toast notification"""
    st.session_state.toast_message = msg
    st.session_state.toast_type = type

def render_toast():
    """Render active toast notification"""
    if st.session_state.toast_message:
        st.markdown(f"""
        <div class="toast toast-{st.session_state.toast_type}">
            {st.session_state.toast_message}
        </div>
        """, unsafe_allow_html=True)
        st.session_state.toast_message = None

def format_overs(overs, balls):
    """Format overs and balls display"""
    return f"{overs}.{balls}"

def calculate_total_balls(overs, balls):
    """Convert overs and balls to total balls"""
    return overs * 6 + balls


# ---------------------------------------------------------
# THEME TOGGLE
# ---------------------------------------------------------
render_toast()

# Theme toggle button in sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    theme_label = "🌙 Dark Mode" if st.session_state.theme == "dark" else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    
    st.divider()
    
    # Reset button in sidebar
    if st.button("🔄 Reset All", use_container_width=True, type="primary"):
        for key, val in defaults.items():
            st.session_state[key] = val
        st.rerun()
    
    # Add cache clear button for troubleshooting
    st.divider()
    st.caption("Troubleshooting")
    if st.button("🔧 Clear Model Cache", use_container_width=True):
        st.cache_resource.clear()
        st.success("Cache cleared! Refreshing...")
        st.rerun()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<div class='title'>🏏 IPL Win Predictor</div>", unsafe_allow_html=True)

# Display selected teams if prediction has been made
if st.session_state.prediction_made and st.session_state.batting_team and st.session_state.bowling_team:
    bat_icon = team_icons.get(st.session_state.batting_team, "🏏")
    bowl_icon = team_icons.get(st.session_state.bowling_team, "🏏")
    st.markdown(f"""
    <div class="vs-header">
        <span>{bat_icon} {st.session_state.batting_team}</span>
        <span class="vs-text">VS</span>
        <span>{bowl_icon} {st.session_state.bowling_team}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

# ==========================
# LEFT COLUMN – INPUT FORM
# ==========================
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Match Configuration")
    
    # Team Selection
    col_bat, col_bowl = st.columns(2)
    
    with col_bat:
        bat = st.selectbox("🏏 Batting Team", teams, key="select_bat")
        st.markdown(f"<div style='text-align:center; font-size:3rem;'>{team_icons[bat]}</div>", 
                   unsafe_allow_html=True)
    
    with col_bowl:
        bowl = st.selectbox("🎯 Bowling Team", teams, key="select_bowl")
        st.markdown(f"<div style='text-align:center; font-size:3rem;'>{team_icons[bowl]}</div>", 
                   unsafe_allow_html=True)
    
    # Validation: Same team check
    if bat == bowl:
        st.warning("⚠️ Batting and bowling teams must be different!")
    
    # Venue Selection
    venue = st.selectbox("📍 Venue", cities, key="select_venue")
    
    st.markdown("---")
    
    # Match Statistics Inputs with improved styling
    st.markdown("#### 📊 Match Statistics")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create 4 columns for inputs - matches HTML grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <label style='font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Current Score</label>
        </div>
        """, unsafe_allow_html=True)
        score_val = st.number_input(
            "Score", 
            min_value=0, 
            value=st.session_state.score,
            key="score_input",
            step=1,
            label_visibility="collapsed"
        )
        st.session_state.score = score_val
        
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <label style='font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Wickets Lost</label>
        </div>
        """, unsafe_allow_html=True)
        wickets_val = st.number_input(
            "Wickets",
            min_value=0,
            max_value=10,
            value=st.session_state.wickets,
            key="wickets_input",
            step=1,
            label_visibility="collapsed"
        )
        st.session_state.wickets = wickets_val
        if wickets_val > 10:
            toast("⚠️ Maximum 10 wickets!", "warning")
    
    with col3:
        st.markdown("""
        <div style='text-align: center;'>
            <label style='font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Target Score</label>
        </div>
        """, unsafe_allow_html=True)
        target_val = st.number_input(
            "Target",
            min_value=1,
            value=st.session_state.target,
            key="target_input",
            step=1,
            label_visibility="collapsed"
        )
        st.session_state.target = target_val
    
    with col4:
        st.markdown("""
        <div style='text-align: center;'>
            <label style='font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Overs</label>
        </div>
        """, unsafe_allow_html=True)
        # Display overs in cricket format (e.g., 14.2)
        overs_display = format_overs(st.session_state.overs, st.session_state.balls)
        st.markdown(f"""
        <div style='text-align: center; background: {T['card']}; padding: 0.75rem; border-radius: 12px; border: 1px solid {T['border']};'>
            <div style='font-size: 2rem; font-weight: 700; color: #3b82f6;'>{overs_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overs and Balls stepper section with better styling
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
        <span class='material-icons' style='color: #9CA3AF;'>schedule</span>
        <h3 style='margin: 0; font-size: 1.25rem; font-weight: 700;'>Overs Completed</h3>
    </div>
    """, unsafe_allow_html=True)
    
    overs_col, balls_col = st.columns(2)
    
    with overs_col:
        st.markdown("""
        <label style='font-size: 0.875rem; font-weight: 500; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Overs</label>
        """, unsafe_allow_html=True)
        
        # Overs value display
        st.markdown(f"""
        <div style='text-align: center; background: {T['card']}; padding: 1rem; border-radius: 8px; border: 1px solid {T['border']}; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.8rem; font-weight: 700; color: {T['text']};'>{st.session_state.overs}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Decrement button
        if st.button("➖ Decrease Overs", key="overs_dec", use_container_width=True):
            if st.session_state.overs > 0:
                st.session_state.overs -= 1
                st.rerun()
        
        # Increment button
        if st.button("➕ Increase Overs", key="overs_inc", use_container_width=True):
            if st.session_state.overs < 20:
                st.session_state.overs += 1
                st.rerun()
            else:
                toast("⚠️ Maximum 20 overs!", "warning")
    
    with balls_col:
        st.markdown("""
        <label style='font-size: 0.875rem; font-weight: 500; color: #9CA3AF; margin-bottom: 0.5rem; display: block;'>Balls</label>
        """, unsafe_allow_html=True)
        
        # Balls value display
        st.markdown(f"""
        <div style='text-align: center; background: {T['card']}; padding: 1rem; border-radius: 8px; border: 1px solid {T['border']}; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.8rem; font-weight: 700; color: {T['text']};'>{st.session_state.balls}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Decrement button
        if st.button("➖ Decrease Balls", key="balls_dec", use_container_width=True):
            if st.session_state.balls > 0:
                st.session_state.balls -= 1
                st.rerun()
        
        # Increment button
        if st.button("➕ Increase Balls", key="balls_inc", use_container_width=True):
            if st.session_state.balls < 5:
                st.session_state.balls += 1
                st.rerun()
            else:
                toast("⚠️ Maximum 5 balls!", "warning")
    
    
    
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    predict_disabled = (bat == bowl) or (pipe is None)
    if st.button("🎯 Predict Win Probability", type="primary", use_container_width=True, disabled=predict_disabled):
        st.session_state.show_loading = True
        st.session_state.prediction_made = False
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# RIGHT COLUMN – RESULTS
# ==========================
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")
    
    # Check if match has ended
    total_balls = calculate_total_balls(st.session_state.overs, st.session_state.balls)
    match_ended = total_balls >= 120 or st.session_state.wickets >= 10
    
    if match_ended and not st.session_state.show_loading:
        st.markdown("""
        <div class="match-ended">
            🏁 Match Ended!
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.score >= st.session_state.target:
            st.success(f"🎉 {bat} wins by {10 - st.session_state.wickets} wickets!")
        else:
            st.error(f"💔 {bat} loses by {st.session_state.target - st.session_state.score} runs!")
    
    # LOADING ANIMATION
    elif st.session_state.show_loading:
        with st.spinner(""):
            st.markdown("<h2 style='text-align:center;'>🏏 Analyzing Match Data...</h2>", 
                       unsafe_allow_html=True)
            time.sleep(1.2)
        
        st.session_state.show_loading = False
        
        # Calculate match statistics
        total_balls_played = calculate_total_balls(st.session_state.overs, st.session_state.balls)
        runs_left = st.session_state.target - st.session_state.score
        balls_left = 120 - total_balls_played
        wickets_left = 10 - st.session_state.wickets
        
        # Calculate rates
        crr = (st.session_state.score * 6) / total_balls_played if total_balls_played > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        # Prepare dataframe for prediction
        df = pd.DataFrame({
            "batting_team": [bat],
            "bowling_team": [bowl],
            "city": [venue],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets": [wickets_left],
            "total_runs_x": [st.session_state.target],
            "crr": [crr],
            "rrr": [rrr]
        })
        
        # Make prediction
        try:
            if pipe is not None:
                prob = pipe.predict_proba(df)[0]
                st.session_state.win = round(prob[1] * 100, 2)
                st.session_state.loss = round(prob[0] * 100, 2)
                st.session_state.batting_team = bat
                st.session_state.bowling_team = bowl
                st.session_state.prediction_made = True
                toast("✅ Prediction successful!", "success")
            else:
                toast("❌ Model not loaded!", "error")
        except Exception as e:
            toast(f"❌ Prediction error: {str(e)}", "error")
        
        st.rerun()
    
    # DISPLAY RESULTS
    elif st.session_state.prediction_made:
        # Recalculate for display (needed for correct values)
        total_balls_played = calculate_total_balls(st.session_state.overs, st.session_state.balls)
        runs_left = st.session_state.target - st.session_state.score
        balls_left = 120 - total_balls_played
        wickets_left = 10 - st.session_state.wickets
        crr = (st.session_state.score * 6) / total_balls_played if total_balls_played > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        # Win Probability Display
        st.markdown("### 🎯 Win Probability")
        
        # Large win percentage display
        st.markdown(f"""
        <div style='text-align:center; margin: 2rem 0;'>
            <div style='font-size: 4rem; font-weight: 800; 
                 background: linear-gradient(135deg, {T['primary']} 0%, {T['secondary']} 100%);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                {st.session_state.win}%
            </div>
            <div style='opacity: 0.7; margin-top: 0.5rem;'>{st.session_state.batting_team} Win Chance</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar visualization
        st.progress(st.session_state.win / 100)
        
        st.markdown("---")
        
        # Team probabilities comparison
        col_win, col_loss = st.columns(2)
        with col_win:
            st.metric(f"🏏 {st.session_state.batting_team}", f"{st.session_state.win}%", 
                     delta=None, delta_color="normal")
        with col_loss:
            st.metric(f"🎯 {st.session_state.bowling_team}", f"{st.session_state.loss}%",
                     delta=None, delta_color="inverse")
        
        st.markdown("---")
        
        # Key Statistics
        st.markdown("#### 📈 Key Statistics")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Required Run Rate</div>
                <div class="metric-value">{rrr:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Runs Needed</div>
                <div class="metric-value">{runs_left}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Run Rate</div>
                <div class="metric-value">{crr:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Balls Remaining</div>
                <div class="metric-value">{balls_left}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Win prediction insight
        if st.session_state.win > 70:
            st.success(f"🎯 Strong advantage for {st.session_state.batting_team}!")
        elif st.session_state.win > 50:
            st.info(f"⚖️ Slight edge to {st.session_state.batting_team}")
        elif st.session_state.win > 30:
            st.warning(f"⚖️ Slight edge to {st.session_state.bowling_team}")
        else:
            st.error(f"🎯 Strong advantage for {st.session_state.bowling_team}!")
    
    else:
        # Initial state - show placeholder
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; opacity: 0.6;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🏏</div>
            <div style='font-size: 1.2rem;'>Configure match details and click<br/><strong>Predict</strong> to see results</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; opacity:0.5; font-size:0.9rem;'>
    Powered by Machine Learning 🤖 | Logistic Regression Model
</div>
""", unsafe_allow_html=True)