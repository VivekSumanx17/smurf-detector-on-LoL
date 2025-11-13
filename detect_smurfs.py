# detect_smurfs.py
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/matches.csv"
MODEL_PATH = "models/rank_clf.joblib"
OUT_CSV = "suspected_teams.csv"

# load teams construction logic (same as training script)
df = pd.read_csv(DATA_PATH)
blue_cols = [c for c in df.columns if c.startswith("blue")]
red_cols = [c for c in df.columns if c.startswith("red")]

blue_team = df[["gameId"] + blue_cols].copy()
red_team = df[["gameId"] + red_cols].copy()
blue_team.columns = ["gameId"] + [c.replace("blue", "") for c in blue_cols]
red_team.columns = ["gameId"] + [c.replace("red", "") for c in red_cols]
blue_team["team"] = "blue"
red_team["team"] = "red"
teams = pd.concat([blue_team, red_team], ignore_index=True)

# recreate score -> rank label exactly as in training
teams["score"] = (
    teams["Kills"] * 2
    + teams["Assists"] * 1.5
    + teams["TowersDestroyed"] * 3
    + teams["TotalGold"] / 1000
    + teams["CSPerMin"] * 1.2
    + teams["EliteMonsters"] * 5
)
teams["rank"] = pd.qcut(teams["score"], 3, labels=["Low", "Medium", "High"])

# features used (should match training features)
feature_cols = [
    "Kills", "Deaths", "Assists",
    "TowersDestroyed", "TotalGold", "AvgLevel", "TotalExperience",
    "TotalMinionsKilled", "TotalJungleMinionsKilled",
    "GoldDiff", "ExperienceDiff", "CSPerMin", "GoldPerMin"
]
X = teams[feature_cols].fillna(0)

# load model
bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
scaler = bundle["scaler"]

X_scaled = scaler.transform(X)
probs = clf.predict_proba(X_scaled)
preds = clf.predict(X_scaled)
class_order = clf.classes_  # e.g., ['Low','Medium','High']

# build report
report = teams[["gameId","team","score","rank"]].copy().reset_index(drop=True)
report["pred"] = preds
# extract probability of predicting 'High' (or better class) — adapt to class ordering
# find index of High in class_order
try:
    high_idx = list(class_order).index("High")
except ValueError:
    # fallback: assume last class is 'High'
    high_idx = len(class_order) - 1

report["prob_high"] = probs[:, high_idx]

# define suspicious condition: labeled Low but model predicts High with high probability
threshold = 0.65
suspected = report[(report["rank"] == "Low") & (report["pred"] == "High") & (report["prob_high"] >= threshold)].copy()
suspected = suspected.sort_values(by="prob_high", ascending=False)

suspected.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(suspected)} suspected overperforming teams (threshold={threshold}).")
print(suspected.head(20))
