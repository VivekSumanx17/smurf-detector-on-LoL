# streamlit_demo.py (styled)
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/rank_clf.joblib"

st.set_page_config(page_title="Smurf Detector — Styled Demo", layout="wide", initial_sidebar_state="expanded")

# ----- small CSS for nicer spacing & fonts -----
st.markdown(
    """
    <style>
    .stApp { font-family: "Segoe UI", Roboto, Arial, sans-serif; }
    .header {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .logo {
        background-color:#0f62fe;
        color: white;
        border-radius:8px;
        padding:10px;
        font-weight:700;
        font-size:18px;
        width:56px;
        height:56px;
        display:flex;
        align-items:center;
        justify-content:center;
    }
    .subtitle { color: #5a6b8a; margin-top: -6px; }
    .metric { padding: 8px 12px; border-radius: 8px; background: #111111; }
    .prob-bar { height: 18px; border-radius: 6px; background: #e9f2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Header -----
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown('<div class="logo">SG</div>', unsafe_allow_html=True)  # SG = Smart Game (your project logo initial)
with col2:
    st.markdown("<div class='header'><h2 style='margin:0'>Smurf Detector</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Team-level early-game tier prediction and suspicious-overperformer flagging</div>", unsafe_allow_html=True)

st.markdown("---")

# ----- Load model -----
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at '{MODEL_PATH}'. Run `python smurf_pipeline.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
clf = bundle.get("model")
scaler = bundle.get("scaler")
class_order = list(clf.classes_)

FEATURE_COLS = [
    "Kills", "Deaths", "Assists",
    "TowersDestroyed", "TotalGold", "AvgLevel", "TotalExperience",
    "TotalMinionsKilled", "TotalJungleMinionsKilled",
    "GoldDiff", "ExperienceDiff", "CSPerMin", "GoldPerMin"
]

# ----- Sidebar controls -----
st.sidebar.header("Demo settings")
prob_threshold = st.sidebar.slider("High probability threshold for suspicious", 0.50, 0.95, 0.65, 0.05)
show_prob_chart = st.sidebar.checkbox("Show probability bar chart", value=True)
batch_download_name = st.sidebar.text_input("Batch download filename", value="predictions.csv")

# ----- Main layout: two columns -----
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Enter team early-game stats")
    with st.form("team_form", clear_on_submit=False):
        # compact grid for inputs
        r1c1, r1c2, r1c3 = st.columns(3)
        inputs = {}
        defaults = {"Kills": 5, "Deaths": 4, "Assists": 8, "TowersDestroyed": 1,
                    "TotalGold": 15000, "AvgLevel": 11.5, "TotalExperience": 18000,
                    "TotalMinionsKilled": 120, "TotalJungleMinionsKilled": 20,
                    "GoldDiff": 100, "ExperienceDiff": 200, "CSPerMin": 20.0, "GoldPerMin": 1600}
        for i, feat in enumerate(FEATURE_COLS):
            col = (r1c1, r1c2, r1c3)[i % 3]
            with col:
                if isinstance(defaults[feat], int):
                    inputs[feat] = st.number_input(feat, value=defaults[feat], step=1, key=f"inp_{feat}")
                else:
                    inputs[feat] = st.number_input(feat, value=float(defaults[feat]), step=0.1, key=f"inp_{feat}_f")
        submitted = st.form_submit_button("Predict this team")

    st.markdown("### Batch upload")
    st.markdown("Upload a CSV with columns matching (or subset of) the feature list.")
    uploaded = st.file_uploader("Upload teams CSV", type=["csv"], accept_multiple_files=False)

with right:
    st.subheader("Prediction")
    # placeholder card
    card = st.empty()

    def render_prediction(X_df):
        X_scaled = scaler.transform(X_df)
        probs = clf.predict_proba(X_scaled)
        preds = clf.predict(X_scaled)
        # show top prediction for first row
        pred0 = preds[0]
        prob0 = probs[0]
        # metrics
        with card.container():
            st.markdown(f"<div class='metric'><strong>Predicted tier:</strong> {pred0}</div>", unsafe_allow_html=True)
            # show probability table
            prob_df = pd.DataFrame({"rank": class_order, "prob": prob0}).sort_values("prob", ascending=False)
            if show_prob_chart:
                st.markdown("**Probabilities**")
                fig, ax = plt.subplots(figsize=(4,1.2))
                ax.barh(prob_df["rank"], prob_df["prob"], height=0.6)
                ax.set_xlim(0,1)
                ax.set_xlabel("probability")
                ax.invert_yaxis()
                ax.set_yticks([])
                for i, (r,p) in enumerate(zip(prob_df["rank"], prob_df["prob"])):
                    ax.text(p + 0.01, i, f"{r}: {p:.2f}", va="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
            st.markdown("**Top probabilities**")
            st.table(prob_df.reset_index(drop=True).head(5))

            # suspicious flag logic: if model predicts High with high prob
            try:
                high_idx = class_order.index("High")
            except ValueError:
                high_idx = len(class_order) - 1
            prob_high = prob0[high_idx]
            if prob_high >= prob_threshold and pred0 == "High":
                st.warning(f"⚠️ Suspicious: model strongly predicts 'High' (prob={prob_high:.2f})")
            else:
                st.success("No strong suspicious flag for this input.")

# ----- Handle single predict submission -----
if submitted:
    X_single = pd.DataFrame([inputs], columns=FEATURE_COLS).fillna(0)
    render_prediction(X_single)

# ----- Handle batch upload -----
if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Failed to read uploaded CSV: " + str(e))
        df_up = None

    if df_up is not None:
        # ensure columns exist
        for c in FEATURE_COLS:
            if c not in df_up.columns:
                df_up[c] = 0
        X_batch = df_up[FEATURE_COLS].fillna(0)
        X_scaled = scaler.transform(X_batch)
        probs = clf.predict_proba(X_scaled)
        preds = clf.predict(X_scaled)
        df_up["predicted_tier"] = preds
        try:
            high_idx = class_order.index("High")
        except ValueError:
            high_idx = len(class_order) - 1
        df_up["prob_high"] = probs[:, high_idx]
        df_up["is_suspicious"] = df_up["prob_high"] >= prob_threshold
        st.markdown("### Batch results (first 20 rows)")
        st.dataframe(df_up.head(20))
        st.markdown(f"Found **{int(df_up['is_suspicious'].sum())}** suspicious rows (threshold = {prob_threshold})")
        csv_bytes = df_up.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv_bytes, batch_download_name, "text/csv")

# ----- Footer notes -----
st.markdown("---")
st.markdown("**Notes:** model trained with `smurf_pipeline.py`. This demo predicts team-level tiers (Low/Medium/High) from early-game aggregated stats.")
st.markdown("Tip: tweak the probability threshold in the sidebar to adjust sensitivity of the suspicious flag.")
