"""
Retrain the IPL Win Predictor model from scratch to fix the pickle serialization issue.

This script recreates the entire training pipeline from the CSV files.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("="*60)
print("IPL Win Predictor - Model Retraining Script")
print("="*60)

# Load data
print("\n1. Loading data from CSV files...")
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
print(f"   Matches: {match.shape}")
print(f"   Deliveries: {delivery.shape}")

# Calculate total runs scored in each first innings
print("\n2. Processing match and delivery data...")
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge with match data
match_df = match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')

# Define active teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Replace and filter teams
print("\n3. Filtering and cleaning team data...")
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# Process delivery data
print("\n4. Processing delivery-level data...")
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Same team name replacements
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]

# Calculate match statistics
print("\n5. Calculating match statistics...")
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])

# Wickets calculation
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets

# Calculate run rates
delivery_df['crr'] = (delivery_df['current_score']*6/(120 - delivery_df['balls_left']))
delivery_df['rrr'] = (delivery_df['runs_left']*6/delivery_df['balls_left'])

# Create result column
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

# Create final dataset
print("\n6. Creating final dataset...")
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
final_df = final_df.sample(final_df.shape[0])
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

print(f"   Final dataset shape: {final_df.shape}")
print(f"   Features: {list(final_df.columns[:-1])}")

# Split into train and test
print("\n7. Splitting data into train and test sets...")
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Create pipeline
print("\n8. Creating machine learning pipeline...")
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
],
remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

print("   Pipeline created:")
print(f"   - Step 1: ColumnTransformer (OneHotEncoder + passthrough)")
print(f"   - Step 2: LogisticRegression")

# Train the model
print("\n9. Training the model...")
pipe.fit(X_train, y_train)
print("   ✅ Training complete!")

# Test the model
print("\n10. Testing the model...")
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print(f"   Training accuracy: {train_score:.4f}")
print(f"   Test accuracy: {test_score:.4f}")

# Test with a sample prediction
print("\n11. Running sample prediction...")
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

prob = pipe.predict_proba(test_df)
print(f"   Test input: MI vs CSK, 65 runs needed off 36 balls")
print(f"   Prediction: {prob[0][1]*100:.2f}% win probability for batting team")

# Check the column transformer
print("\n12. Verifying ColumnTransformer structure...")
ct = pipe.steps[0][1]
print(f"   Remainder type: {type(ct.remainder)}")
print(f"   Transformers_:")
for name, transformer, columns in ct.transformers_:
    print(f"     - {name}: {type(transformer).__name__}")

# Save the model
print("\n13. Saving the model...")
import shutil
import os

if os.path.exists('pipe.pkl'):
    shutil.copy('pipe.pkl', 'pipe.pkl.backup')
    print("   ✅ Created backup: pipe.pkl.backup")

pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("   ✅ Saved new model to pipe.pkl")

# Verify  the saved model
print("\n14. Verifying saved model...")
loaded_pipe = pickle.load(open('pipe.pkl', 'rb'))
prob_verify = loaded_pipe.predict_proba(test_df)
print(f"   ✅ Loaded model prediction: {prob_verify[0][1]*100:.2f}%")

print("\n" + "="*60)
print("✅ SUCCESS! Model has been retrained and saved.")
print("="*60)
print("\nModel Statistics:")
print(f"  - Training samples: {X_train.shape[0]:,}")
print(f"  - Test accuracy: {test_score:.4f}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Teams: {len(teams)}")
print(f"  - Cities: {len(delivery_df['city'].unique())}")
print("\nThe model is ready to use!")
