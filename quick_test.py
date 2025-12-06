"""
Quick test to verify the running Streamlit app is using the correct model.
This simulates what happens when a user makes a prediction through the web interface.
"""
import pickle
import pandas as pd

print("Testing the model file that Streamlit is loading...")
print("="*70)

# Load the model the same way Streamlit does
with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

# Check the model structure
ct = pipe.steps[0][1]
print(f"\nColumnTransformer remainder: {type(ct.remainder).__name__} = {repr(ct.remainder)}")
print(f"Transformers_:")
for name, transformer, columns in ct.transformers_:
    print(f"  - {name}: {type(transformer).__name__}")

# Test prediction
test_data = pd.DataFrame({
    "batting_team": ["Mumbai Indians"],
    "bowling_team": ["Chennai Super Kings"],
    "city": ["Mumbai"],
    "runs_left": [65],
    "balls_left": [36],
    "wickets": [7],
    "total_runs_x": [189],
    "crr": [8.17],
    "rrr": [10.83]
})

print("\nTesting prediction...")
try:
    prob = pipe.predict_proba(test_data)
    print(f"✅ SUCCESS! Prediction: {prob[0][1]*100:.2f}% win probability")
    print("\nThe model is working correctly!")
    print("Your Streamlit app should now work without errors.")
    print(f"\nAccess it at: http://localhost:8501")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
