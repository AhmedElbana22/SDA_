import pandas as pd
import numpy as np
import json
import requests
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.tokenize import word_tokenize
from groq import Groq
from main import clean_text, apply_lemmatization, remove_stopwords

df = pd.read_csv('Movies.csv')
df = df.head(50)    
print(f"Loaded {len(df)} records\n")

# ground truth labeling 

PROMPTS = [
    # Prompt A – direct polarity question
    lambda text: (
        f"Classify the sentiment of this movie review as exactly one word: "
        f"positive, negative, or neutral.\n\nReview: {text}\n\nSentiment:"
    ),
    # Prompt B – audience emotion angle
    lambda text: (
        f"How would a typical viewer feel after reading this review? "
        f"Reply with exactly one word: positive, negative, or neutral.\n\n"
        f"Review: {text}\n\nFeeling:"
    ),
    # Prompt C – critic perspective
    lambda text: (
        f"From a film-critic perspective, is the following review positive, "
        f"negative, or neutral? Reply with that single word only.\n\n"
        f"Review: {text}\n\nVerdict:"
    ),
]


client = Groq(api_key="your_api_key_here")  # replace with your actual API key

def groq_label(text: str, prompt_fn) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_fn(text)}],
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip().lower()
    for label in ("positive", "negative", "neutral"):
        if label in raw:
            return label
    return "neutral"              


def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add annotator_1/2/3 columns via three Groq prompts."""
    labels = {f"annotator_{i+1}": [] for i in range(3)}
    for text in df["Review_Text"].fillna(""):
        for i, prompt_fn in enumerate(PROMPTS):
            labels[f"annotator_{i+1}"].append(groq_label(text, prompt_fn))
    for col, vals in labels.items():
        df[col] = vals
    return df

df = annotate_dataframe(df)
 
# fleiss_kappa function

def fleiss_kappa(ratings: np.ndarray) -> float:

    N, k = ratings.shape
    n = ratings.sum(axis=1)[0]            # raters per item
    p_j = ratings.sum(axis=0) / (N * n)  # category prop
    P_e = (p_j ** 2).sum()
    P_i = ((ratings ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    kappa = (P_bar - P_e) / (1 - P_e)
    return round(float(kappa), 4)


CATEGORIES = ["positive", "negative", "neutral"]

def build_rating_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = ["annotator_1", "annotator_2", "annotator_3"]
    mat = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    for i, row in df.iterrows():
        for col in cols:
            label = row[col]
            if label in CATEGORIES:
                mat[i, CATEGORIES.index(label)] += 1
    return mat

rating_mat = build_rating_matrix(df)
print("Fleiss Kappa:", fleiss_kappa(rating_mat))

# Majority vote for ground truth

def majority_vote(row) -> str:
    votes = [row[f"annotator_{i+1}"] for i in range(3)]
    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "neutral"
    return top[0][0]

df["ground_truth"] = df.apply(majority_vote, axis=1)
df = df[df["ground_truth"].notna()]
print("\nGround Truth distribution:")
print(df["ground_truth"].value_counts())

df.to_csv("Labeled_Movies.csv", index=False)
print("Saved Labeled_Movies.csv\n")

# Text preprocessing

df["clean_text"] = df["Review_Text"].fillna("").apply(clean_text)

# Scheme 1 – basic cleaning only
texts1 = df["clean_text"].tolist()
# Scheme 2 – cleaning + stopword removal
texts2 = [remove_stopwords(t) for t in texts1]
# Scheme 3 – cleaning + stopword removal + lemmatization
texts3 = [apply_lemmatization(remove_stopwords(t)) for t in texts1]

y = np.array(df["ground_truth"])

# Text representation

# BOW
def build_bow(texts, label="") -> tuple:
    vec = CountVectorizer(max_features=500, ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)
    return matrix, vec

bow1, vec_bow1 = build_bow(texts1, "scheme1")
bow2, vec_bow2 = build_bow(texts2, "scheme2")
bow3, vec_bow3 = build_bow(texts3, "scheme3")

with open("bow_representation.json", "w") as f:
    json.dump({
        "scheme": "Scheme 3 (clean + stopwords + lemma)",
        "feature_names": vec_bow3.get_feature_names_out().tolist()[:20],
        "sample_vectors": bow3[:5].toarray().tolist(),
    }, f, indent=2)
print("Saved bow_representation.json")

# ── 6B  GloVe  (real glove.6B.100d.txt embeddings – 100 dimensions)
def load_glove(path: str = "glove.6B.100d.txt") -> dict:
    """Read pre-trained GloVe vectors into a word→vector dict."""
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings):,} GloVe vectors (dim=100)")
    return embeddings

GLOVE = load_glove("glove.6B.100d.txt")
GLOVE_DIM = 100

def text_to_glove(text: str) -> np.ndarray:
    """Average GloVe vectors for all known words in the text."""
    tokens = text.split()
    vecs = [GLOVE[w] for w in tokens if w in GLOVE]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(GLOVE_DIM, dtype=np.float32)

glove1 = np.vstack([text_to_glove(t) for t in texts1])
glove2 = np.vstack([text_to_glove(t) for t in texts2])
glove3 = np.vstack([text_to_glove(t) for t in texts3])

with open("glove_representation.json", "w") as f:
    json.dump({
        "note": "Real GloVe glove.6B.100d – 100 dims, averaged per document, Scheme 3",
        "shape": list(glove3.shape),
        "sample_vectors": glove3[:5].tolist(),
    }, f, indent=2)
print("Saved glove_representation.json\n")

# Lexical Models 

nltk.download("sentiwordnet", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# SentiWordNet classifier
def get_pos_tag(tag: str) -> str | None:
    """Map Penn Treebank POS tag to WordNet POS."""
    if tag.startswith("J"):  return wn.ADJ
    if tag.startswith("V"):  return wn.VERB
    if tag.startswith("N"):  return wn.NOUN
    if tag.startswith("R"):  return wn.ADV
    return None

def sentiwordnet_label(text: str) -> str:
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    pos_score, neg_score = 0.0, 0.0
    for word, tag in tagged:
        wn_tag = get_pos_tag(tag)
        if wn_tag is None:
            continue
        synsets = list(swn.senti_synsets(word, wn_tag))
        if synsets:
            ss = synsets[0]              
            pos_score += ss.pos_score()
            neg_score += ss.neg_score()
    if pos_score > neg_score:   return "positive"
    elif neg_score > pos_score: return "negative"
    return "neutral"

# Bing Liu dictionary classifier with negation handling
def load_opinion_words(url: str) -> set:
    """Fetch Bing Liu positive/negative word list and return as a set."""
    resp = requests.get(url, timeout=15)
    words = set()
    for line in resp.text.splitlines():
        line = line.strip()
        if line and not line.startswith(";"):   
            words.add(line.lower())
    return words

POSITIVE_WORDS = load_opinion_words("https://ptrckprry.com/course/ssd/data/positive-words.txt")
NEGATIVE_WORDS = load_opinion_words("https://ptrckprry.com/course/ssd/data/negative-words.txt")
print(f"Bing Liu: {len(POSITIVE_WORDS)} positive, {len(NEGATIVE_WORDS)} negative words loaded")

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere",
    "hardly", "barely", "scarcely", "without", "cant", "cannot", "wont",
    "dont", "doesnt", "didnt", "wasnt", "isnt", "aint", "couldnt", "shouldnt",
}

def bing_liu_label(text: str) -> str:
    """Bing Liu dictionary classifier with negation handling (window = 3 words)."""
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    words = text.split()
    pos, neg = 0, 0
    for i, word in enumerate(words):
        # check negation in a 3-word window before current word
        negated = any(w in NEGATION_WORDS for w in words[max(0, i - 3):i])
        if word in POSITIVE_WORDS:
            neg += negated; pos += not negated
        elif word in NEGATIVE_WORDS:
            pos += negated; neg += not negated
    if pos > neg:   return "positive"
    elif neg > pos: return "negative"
    return "neutral"

# Evaluate lexical models
print("\n── SentiWordNet ──")
swn_pred = np.array([sentiwordnet_label(t) for t in texts1])
print("Accuracy:", round(accuracy_score(y, swn_pred), 4))
print(classification_report(y, swn_pred, zero_division=0))

print("\n── Bing Liu (with negation) ──")
bing_pred = np.array([bing_liu_label(t) for t in texts1])
print("Accuracy:", round(accuracy_score(y, bing_pred), 4))
print(classification_report(y, bing_pred, zero_division=0))

# ML models 

def run_ml_models(X, y_labels: np.ndarray, name: str):
    print(f"\n── {name} ──")
    if len(np.unique(y_labels)) < 2:
        print("  Not enough classes – skipping.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # Naive Bayes
    if hasattr(X, "toarray"):
        # Sparse (BoW) → MultinomialNB (non-negative counts)
        nb = MultinomialNB()
    else:
        # Dense (GloVe) → GaussianNB (handles negative values)
        nb = GaussianNB()

    nb.fit(X_tr, y_tr)
    nb_pred = nb.predict(X_te)
    print("  Naive Bayes  Accuracy:", round(accuracy_score(y_te, nb_pred), 4))
    print(classification_report(y_te, nb_pred, zero_division=0))

    # Decision Tree (works for both)
    dt = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
    dt.fit(X_tr, y_tr)
    dt_pred = dt.predict(X_te)
    print("  Decision Tree  Accuracy:", round(accuracy_score(y_te, dt_pred), 4))
    print(classification_report(y_te, dt_pred, zero_division=0))


# BoW × 3 schemes
run_ml_models(bow1, y, "Scheme 1  BoW")
run_ml_models(bow2, y, "Scheme 2  BoW")
run_ml_models(bow3, y, "Scheme 3  BoW")

# GloVe × 3 schemes  (dense arrays → ComplementNB / DT)
run_ml_models(glove1, y, "Scheme 1  GloVe")
run_ml_models(glove2, y, "Scheme 2  GloVe")
run_ml_models(glove3, y, "Scheme 3  GloVe")

# Binary (positive vs negative, drop neutral)
mask = y != "neutral"
run_ml_models(bow1[mask],   y[mask], "Scheme 1  BoW   Binary")
run_ml_models(bow2[mask],   y[mask], "Scheme 2  BoW   Binary")
run_ml_models(bow3[mask],   y[mask], "Scheme 3  BoW   Binary")
run_ml_models(glove1[mask], y[mask], "Scheme 1  GloVe Binary")
run_ml_models(glove2[mask], y[mask], "Scheme 2  GloVe Binary")
run_ml_models(glove3[mask], y[mask], "Scheme 3  GloVe Binary")

print("\nDONE")
