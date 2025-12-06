import pickle
import pandas as pd

# Load the pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Create test data
bat = "Mumbai Indians"
bowl = "Chennai Super Kings"
venue = "Mumbai"
runs_left = 65
balls_left = 36
wickets_left = 7
target = 189
crr = 8.17
rrr = 10.83

# Prepare dataframe for prediction
df = pd.DataFrame({
    "batting_team": [bat],
    "bowling_team": [bowl],
    "city": [venue],
    "runs_left": [runs_left],
    "balls_left": [balls_left],
    "wickets": [wickets_left],
    "total_runs_x": [target],
    "crr": [crr],
    "rrr": [rrr]
})

print("Input DataFrame:")
print(df)
print("\nData types:")
print(df.dtypes)

try:
    print("\nAttempting prediction...")
    prob = pipe.predict_proba(df)
    print(f"Prediction successful: {prob}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
