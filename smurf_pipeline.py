import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/matches.csv"
MODEL_OUT = "models/rank_clf.joblib"
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- Step 1: Build team-level rows ---
blue_cols = [c for c in df.columns if c.startswith("blue")]
red_cols = [c for c in df.columns if c.startswith("red")]

blue_team = df[["gameId"] + blue_cols].copy()
red_team = df[["gameId"] + red_cols].copy()

# rename columns to unify names
blue_team.columns = ["gameId"] + [c.replace("blue", "") for c in blue_cols]
red_team.columns = ["gameId"] + [c.replace("red", "") for c in red_cols]

# add label column
blue_team["team"] = "blue"
red_team["team"] = "red"

# combine both teams
teams = pd.concat([blue_team, red_team], ignore_index=True)

# --- Step 2: Create a pseudo "rank"/skill label ---
# We'll define skill tiers based on early performance proxies (gold per min, CS per min, kills, etc.)
# Compute an overall "performance score" for each team.
teams["score"] = (
    teams["Kills"] * 2
    + teams["Assists"] * 1.5
    + teams["TowersDestroyed"] * 3
    + teams["TotalGold"] / 1000
    + teams["CSPerMin"] * 1.2
    + teams["EliteMonsters"] * 5
)

# Label skill tiers by dividing score into 3 bins: Low, Medium, High
teams["rank"] = pd.qcut(teams["score"], 3, labels=["Low", "Medium", "High"])

print("Created synthetic rank labels based on performance.")
print(teams[["team", "score", "rank"]].head())

# --- Step 3: Select features ---
feature_cols = [
    "Kills", "Deaths", "Assists",
    "TowersDestroyed", "TotalGold", "AvgLevel", "TotalExperience",
    "TotalMinionsKilled", "TotalJungleMinionsKilled",
    "GoldDiff", "ExperienceDiff", "CSPerMin", "GoldPerMin"
]

X = teams[feature_cols].fillna(0)
y = teams["rank"]

# --- Step 4: Train/test split ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 5: Train model ---
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Low","Medium","High"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Low","Medium","High"], yticklabels=["Low","Medium","High"])
plt.title("Confusion Matrix (rows = true, cols = predicted)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# --- Step 7: Feature importance ---
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(7,4))
importances.plot(kind="bar")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("Saved feature_importances.png")

# --- Step 8: Save model ---
joblib.dump({"model": clf, "scaler": scaler}, MODEL_OUT)
print("Saved model:", MODEL_OUT)

print("\n✅ Training complete! You now have a working supervised model classifying team performance tiers.")
