"""
Fix the pipe.pkl model file by properly rebuilding the ColumnTransformer.

This script loads the existing model, extracts the trained parameters,
and rebuilds it with a properly initialized ColumnTransformer.
"""
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("Loading existing model...")
with open('pipe.pkl', 'rb') as f:
    old_pipe = pickle.load(f)

print("Old pipeline structure:")
print(f"  Steps: {[name for name, _ in old_pipe.steps]}")

# Extract the old column transformer and logistic regression
old_ct = old_pipe.steps[0][1]
old_lr = old_pipe.steps[1][1]

print("\nOld ColumnTransformer:")
print(f"  Remainder type: {type(old_ct.remainder)}")
print(f"  Remainder value: {repr(old_ct.remainder)}")

# Get the fitted OneHotEncoder from the old model
old_ohe = old_ct.transformers_[0][1]

print("\nOld OneHotEncoder categories:")
for i, cat in enumerate(old_ohe.categories_):
    print(f"  Feature {i}: {len(cat)} categories")

# Create a new ColumnTransformer with proper configuration
# We'll use FunctionTransformer('passthrough') instead of string 'passthrough'
from sklearn.preprocessing import FunctionTransformer

new_ohe = OneHotEncoder(sparse_output=False, drop='first')
# Manually set the categories from the old model
new_ohe.categories_ = old_ohe.categories_
new_ohe.drop_idx_ = old_ohe.drop_idx_
new_ohe._drop_idx_after_grouping = old_ohe._drop_idx_after_grouping
new_ohe.feature_names_in_ = old_ohe.feature_names_in_ if hasattr(old_ohe, 'feature_names_in_') else None
new_ohe.n_features_in_ = old_ohe.n_features_in_
new_ohe._n_features_outs = old_ohe._n_features_outs if hasattr(old_ohe, '_n_features_outs') else None

print("\nCreating new ColumnTransformer...")
new_ct = ColumnTransformer([
    ('trf', new_ohe, ['batting_team', 'bowling_team', 'city'])
],
remainder='passthrough')

# Manually set the fitted attributes
new_ct.transformers_ = [
    ('trf', new_ohe, ['batting_team', 'bowling_team', 'city']),
    ('remainder', FunctionTransformer(), [3, 4, 5, 6, 7, 8])  # Use FunctionTransformer instead of string
]
new_ct._columns = old_ct._columns
new_ct._remainder = old_ct._remainder
new_ct._n_features = old_ct._n_features
new_ct.sparse_output_ = old_ct.sparse_output_
new_ct._feature_names_in = old_ct._feature_names_in if hasattr(old_ct, '_feature_names_in') else None
new_ct.n_features_in_ = old_ct.n_features_in_
new_ct.feature_names_in_ = old_ct.feature_names_in_ if hasattr(old_ct, 'feature_names_in_') else None
new_ct.output_indices_ = old_ct.output_indices_

print("\nCreating new pipeline...")
new_pipe = Pipeline(steps=[
    ('step1', new_ct),
    ('step2', old_lr)  # Reuse the same trained logistic regression
])

print("\nNew pipeline structure:")
print(f"  Steps: {[name for name, _ in new_pipe.steps]}")

print("\nNew ColumnTransformer:")
new_ct_check = new_pipe.steps[0][1]
print(f"  Transformers_:")
for name, transformer, columns in new_ct_check.transformers_:
    print(f"    {name}: {type(transformer).__name__} on columns: {columns}")

# Test the new pipeline
print("\n" + "="*60)
print("Testing new pipeline...")
print("="*60)

import pandas as pd

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

print("\nTest input:")
print(test_df)

try:
    prob = new_pipe.predict_proba(test_df)
    print(f"\n✅ Prediction successful!")
    print(f"Probabilities: {prob}")
    print(f"Win probability: {prob[0][1]*100:.2f}%")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save the new pipeline
print("\n" + "="*60)
print("Saving fixed pipeline to pipe.pkl...")
print("="*60)

# Backup the old file first
import shutil
import os
if os.path.exists('pipe.pkl'):
    shutil.copy('pipe.pkl', 'pipe.pkl.backup')
    print("✅ Created backup: pipe.pkl.backup")

with open('pipe.pkl', 'wb') as f:
    pickle.dump(new_pipe, f)

print("✅ Saved new pipeline to pipe.pkl")

# Verify the saved file
print("\nVerifying saved file...")
with open('pipe.pkl', 'rb') as f:
    verified_pipe = pickle.load(f)

prob_verify = verified_pipe.predict_proba(test_df)
print(f"✅ Verification successful!")
print(f"Probabilities: {prob_verify}")

print("\n" + "="*60)
print("SUCCESS! Model has been fixed and saved.")
print("="*60)
