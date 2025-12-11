from ml_pipeline.predict import predict_next_games
import pandas as pd

print("Testing predict_next_games() output...")
df = predict_next_games()

print(f"\n=== DataFrame Info ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\n=== Checking Injury Columns ===")
if 'home_injuries_list' in df.columns:
    print("✅ home_injuries_list EXISTS")
    print(f"Sample values:\n{df[['home_team', 'home_injuries_list']].head()}")
else:
    print("❌ home_injuries_list MISSING")

if 'away_injuries_list' in df.columns:
    print("✅ away_injuries_list EXISTS")
    print(f"Sample values:\n{df[['away_team', 'away_injuries_list']].head()}")
else:
    print("❌ away_injuries_list MISSING")

print(f"\n=== Converting to dict (como cli.py faz) ===")
sample_dict = df.iloc[0].to_dict()
print(f"Keys: {list(sample_dict.keys())}")
print(f"home_injuries_list value: {sample_dict.get('home_injuries_list', 'KEY NOT FOUND')}")
print(f"away_injuries_list value: {sample_dict.get('away_injuries_list', 'KEY NOT FOUND')}")
