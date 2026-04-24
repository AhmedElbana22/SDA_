import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from main import clean_text, apply_lemmatization, remove_stopwords

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── Load Best Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_best_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    data        = load_best_model()
    model       = data["model"]
    model_name  = data["model_name"]
    macro_f1    = data["macro_f1"]
    cv_f1       = data["cv_f1"]
    is_glove    = data["is_glove"]
    vectorizer  = data["vectorizer"]
    GLOVE       = data["glove"]
    scheme      = data["scheme"]
    classes     = data["classes"]
    GLOVE_DIM   = 100
except FileNotFoundError:
    st.error("⚠️ best_model.pkl not found. Please run optimization.py first.")
    st.stop()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text):
    cleaned = clean_text(text)
    if "Scheme1" in scheme:
        return cleaned
    elif "Scheme2" in scheme:
        return remove_stopwords(cleaned)
    else:
        return apply_lemmatization(remove_stopwords(cleaned))

def text_to_glove(text):
    tokens = text.split()
    vecs = [GLOVE[w] for w in tokens if w in GLOVE]
    return np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, GLOVE_DIM), dtype=np.float32)

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(text):
    processed = preprocess(text)
    X = text_to_glove(processed) if is_glove else vectorizer.transform([processed])
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(X)[0]
        conf_dict = {c: round(float(p), 4) for c, p in zip(model.classes_, proba)}
    else:
        conf_dict = {pred: 1.0}
    return pred, conf_dict

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Analyze the sentiment of any movie review using our best trained model.")

# Model info banner
st.info(
    f"**Model:** `{model_name}`  \n"
    f"**Macro F1:** `{macro_f1}`  |  **CV F1 (5-fold):** `{cv_f1}`  \n"
    f"**Representation:** `{'GloVe' if is_glove else 'BOW'}`  |  **Scheme:** `{scheme}`"
)

st.markdown("---")

# Input
review_text = st.text_area(
    "📝 Enter a movie review:",
    placeholder="e.g. This movie was absolutely fantastic! The acting was superb and the story kept me engaged throughout...",
    height=160
)

predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

# ── Result ─────────────────────────────────────────────────────────────────────
if predict_btn:
    if not review_text.strip():
        st.warning("Please enter a movie review first.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence = predict(review_text)

        color_map = {
            "positive": ("#2ecc71", "😊 POSITIVE"),
            "negative": ("#e74c3c", "😞 NEGATIVE"),
            "neutral":  ("#f39c12", "😐 NEUTRAL"),
        }
        color, label = color_map.get(sentiment, ("#95a5a6", "🤔 UNKNOWN"))
        top_conf = confidence.get(sentiment, 0)

        st.markdown("---")

        # Big result display
        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{label}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center; font-size:18px;'>Confidence: <b>{round(top_conf*100, 1)}%</b></p>",
            unsafe_allow_html=True
        )

        # Pie chart
        pie_colors = [color_map.get(c, ("#95a5a6", ""))[0] for c in confidence.keys()]
        fig = go.Figure(data=[go.Pie(
            labels=list(confidence.keys()),
            values=[round(v * 100, 2) for v in confidence.values()],
            hole=0.45,
            marker=dict(colors=pie_colors),
            textinfo="label+percent",
            textfont_size=15,
        )])
        fig.update_layout(
            title_text="Sentiment Distribution",
            showlegend=True,
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # JSON response
        st.markdown("**API Response (JSON):**")
        st.json({"sentiment": sentiment, "confidence": round(top_conf, 4)})

st.markdown("---")
st.caption("Social Data Analytics Project — Movie Sentiment Analysis")