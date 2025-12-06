import pickle
import pandas as pd
import json

# Load the pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Simulate data coming from JSON (like from a mobile app or web request)
json_data = json.dumps({
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "city": "Mumbai",
    "runs_left": 65,
    "balls_left": 36,
    "wickets": 7,
    "total_runs_x": 189,
    "crr": 8.17,
    "rrr": 10.83
})

# Parse it back (simulating what happens in production)
data = json.loads(json_data)

# Create DataFrame - this is how the app would do it
df = pd.DataFrame([data])

print("Input DataFrame:")
print(df)
print("\nData types:")
print(df.dtypes)
print("\nFirst row values:")
print(df.iloc[0])

try:
    print("\nAttempting prediction...")
    prob = pipe.predict_proba(df)
    print(f"Prediction successful: {prob}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    
    # Check the transformers in detail
    print("\n\n=== DEBUGGING ===")
    ct = pipe.steps[0][1]
    print("ColumnTransformer transformers_:")
    for name, transformer, columns in ct.transformers_:
        print(f"\n  {name}: {type(transformer)}")
        print(f"    Transformer type: {transformer}")
        print(f"    Columns: {columns}")
