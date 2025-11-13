# generate_synthetic.py
import pandas as pd
import numpy as np
import random
import os

os.makedirs("data", exist_ok=True)
n_matches = 5000  # change as needed
ranks = ["Iron","Bronze","Silver","Gold","Platinum","Diamond","Master","Grandmaster","Challenger"]

def sample_rank():
    return random.choices(ranks, weights=[5,10,20,30,18,10,4,2,1])[0]

rows = []
for i in range(n_matches):
    match_id = f"M{i:05d}"
    for slot in range(10):
        kills = np.random.poisson(1.2)
        deaths = np.random.poisson(1.0)
        assists = np.random.poisson(2.0)
        cs_per_min = round(max(0, np.random.normal(6.0, 2.0)),2)
        gold_per_min = round(max(200, np.random.normal(350, 50)))
        vision_score = max(0, int(np.random.normal(20,10)))
        xp_per_min = round(max(200, np.random.normal(300, 40)))
        rank = sample_rank()
        rows.append({
            "match_id": match_id,
            "player_id": f"P{random.randint(1000,9999)}",
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "cs_per_min": cs_per_min,
            "gold_per_min": gold_per_min,
            "vision_score": vision_score,
            "xp_per_min": xp_per_min,
            "rank": rank
        })

df = pd.DataFrame(rows)
df.to_csv("data/synthetic_players.csv", index=False)
print("Saved data/synthetic_players.csv with shape", df.shape)
