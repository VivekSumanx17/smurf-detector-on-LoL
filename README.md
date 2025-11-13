# Smurf Detector for League of Legends (LoL)
An end-to-end **machine learning project** that detects **smurf / overperforming players** in **League of Legends** using **supervised learning** and an interactive **Streamlit dashboard**.

This project predicts team skill tiers (Low / Medium / High) from early-game statistics and flags teams whose performance is significantly higher than expected — a common indicator of smurfing.

---

# Features

# Machine Learning Pipeline
- Supervised learning (Random Forest Classifier)
- Early-game feature extraction from LoL match data
- Automatic skill tier prediction (Low / Medium / High)
- Feature importance visualizations
- Confusion matrix for evaluating model accuracy
- Synthetic dataset support if real ranks are missing

# Smurf Detection Engine
- Detects **overperforming teams** based on:
  - model prediction,
  - prediction probability,
  - difference from expected tier.
- Exports flagged teams to CSV.

# Streamlit Web App
- Clean, styled UI
- Single-team prediction form
- Batch CSV upload
- Probability visualizations
- Suspicious (smurf-like) flagging system

---

# ML Approach

This project uses **Supervised Machine Learning → Classification**  
Algorithm used:

# Random Forest Classifier
Chosen because:
- handles nonlinear game stats well  
- robust against noise  
- interpretable (feature importances)  
- works well with small/medium datasets  

---

## 📂 Project Structure

