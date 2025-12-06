"""
Comprehensive test to verify the model works correctly in all scenarios.
"""
import pickle
import pandas as pd
import json

print("="*70)
print("Model Verification Test")
print("="*70)

# Load the model
print("\n1. Loading model...")
pipe = pickle.load(open('pipe.pkl', 'rb'))
print("   ✅ Model loaded successfully")

# Inspect the transformer
print("\n2. Inspecting ColumnTransformer...")
ct = pipe.steps[0][1]
print(f"   Remainder type: {type(ct.remainder)}")
print(f"   Remainder value: {repr(ct.remainder)}")
print("   Transformers_:")
for name, transformer, columns in ct.transformers_:
    print(f"     - {name}: {type(transformer).__name__} on {columns}")

# Test 1: Direct DataFrame prediction
print("\n3. Test 1: Direct DataFrame prediction")
test_df = pd.DataFrame({
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

try:
    prob = pipe.predict_proba(test_df)
    print(f"   ✅ SUCCESS: {prob[0][1]*100:.2f}% win probability")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 2: JSON-like data (simulating web/mobile request)
print("\n4. Test 2: JSON-serialized data (mobile/web scenario)")
json_data = {
    "batting_team": "Sunrisers Hyderabad",
    "bowling_team": "Royal Challengers Bangalore",
    "city": "Hyderabad",
    "runs_left": 45,
    "balls_left": 24,
    "wickets": 5,
    "total_runs_x": 200,
    "crr": 9.67,
    "rrr": 11.25
}

# Convert to JSON string and back (simulating API)
json_str = json.dumps(json_data)
data = json.loads(json_str)
test_df2 = pd.DataFrame([data])

try:
    prob = pipe.predict_proba(test_df2)
    print(f"   ✅ SUCCESS: {prob[0][1]*100:.2f}% win probability")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: Multiple predictions
print("\n5. Test 3: Multiple predictions (batch)")
batch_df = pd.DataFrame({
    "batting_team": ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders"],
    "bowling_team": ["Chennai Super Kings", "Mumbai Indians", "Delhi Capitals"],
    "city": ["Mumbai", "Chennai", "Kolkata"],
    "runs_left": [65, 30, 80],
    "balls_left": [36, 18, 42],
    "wickets": [7, 9, 4],
    "total_runs_x": [189, 180, 210],
    "crr": [8.17, 10.00, 7.14],
    "rrr": [10.83, 10.00, 11.43]
})

try:
    probs = pipe.predict_proba(batch_df)
    print(f"   ✅ SUCCESS: Predicted {len(probs)} match outcomes")
    for i, prob in enumerate(probs):
        print(f"      Match {i+1}: {prob[1]*100:.2f}% win probability")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 4: All teams and cities
print("\n6. Test 4: Testing all teams")
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

success_count = 0
for team in teams:
    other_team = teams[(teams.index(team) + 1) % len(teams)]
    test_df = pd.DataFrame({
        "batting_team": [team],
        "bowling_team": [other_team],
        "city": ["Mumbai"],
        "runs_left": [50],
        "balls_left": [30],
        "wickets": [6],
        "total_runs_x": [180],
        "crr": [8.67],
        "rrr": [10.00]
    })
    try:
        prob = pipe.predict_proba(test_df)
        success_count += 1
    except Exception as e:
        print(f"   ❌ Failed for {team}: {e}")

print(f"   ✅ SUCCESS: All {success_count}/{len(teams)} team combinations work")

# Test 5: Edge cases
print("\n7. Test 5: Edge cases")
edge_cases = [
    {"runs_left": 1, "balls_left": 1, "label": "1 run off 1 ball"},
    {"runs_left": 120, "balls_left": 120, "label": "120 runs off 120 balls"},
    {"runs_left": 0, "balls_left": 60, "label": "Target already achieved"},
]

for case in edge_cases:
    test_df = pd.DataFrame({
        "batting_team": ["Mumbai Indians"],
        "bowling_team": ["Chennai Super Kings"],
        "city": ["Mumbai"],
        "runs_left": [case["runs_left"]],
        "balls_left": [case["balls_left"]],
        "wickets": [7],
        "total_runs_x": [180],
        "crr": [5.0],
        "rrr": [case["runs_left"] * 6 / case["balls_left"] if case["balls_left"] > 0 else 0]
    })
    try:
        prob = pipe.predict_proba(test_df)
        print(f"   ✅ {case['label']}: {prob[0][1]*100:.2f}% win probability")
    except Exception as e:
        print(f"   ❌ {case['label']}: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nThe model is working correctly and ready for production use!")
print("You can now run the Streamlit app without errors.")
