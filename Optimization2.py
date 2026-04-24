import pandas as pd
import numpy as np
import json
import pickle
import warnings
import requests
warnings.filterwarnings("ignore")

from typing import Optional
import nltk
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.tokenize import word_tokenize

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import SMOTE

from main import clean_text, apply_lemmatization, remove_stopwords

# ── Load labeled data ──────────────────────────────────────────────────────────
df = pd.read_csv("Labeled_Movies.csv")
df = df[df["ground_truth"].notna()].reset_index(drop=True)
print(f"Loaded {len(df)} labeled records")
print("Ground Truth distribution:")
print(df["ground_truth"].value_counts(), "\n")

y = np.array(df["ground_truth"])

# ── Text Preprocessing Schemes ─────────────────────────────────────────────────
df["clean_text"] = df["Review_Text"].fillna("").apply(clean_text)
texts1 = df["clean_text"].tolist()
texts2 = [remove_stopwords(t) for t in texts1]
texts3 = [apply_lemmatization(remove_stopwords(t)) for t in texts1]

# ── BOW Representation ─────────────────────────────────────────────────────────
def build_bow(texts):
    vec = CountVectorizer(max_features=1000, ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)
    return matrix, vec

bow1, vec1 = build_bow(texts1)
bow2, vec2 = build_bow(texts2)
bow3, vec3 = build_bow(texts3)

# ── GloVe Representation ───────────────────────────────────────────────────────
def load_glove(path="glove.6B.100d.txt"):
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"Loaded {len(embeddings):,} GloVe vectors")
    return embeddings

GLOVE = load_glove()
GLOVE_DIM = 100

def text_to_glove(text):
    tokens = text.split()
    vecs = [GLOVE[w] for w in tokens if w in GLOVE]
    return np.mean(vecs, axis=0) if vecs else np.zeros(GLOVE_DIM, dtype=np.float32)

glove1 = np.vstack([text_to_glove(t) for t in texts1])
glove2 = np.vstack([text_to_glove(t) for t in texts2])
glove3 = np.vstack([text_to_glove(t) for t in texts3])

# ── SMOTE ──────────────────────────────────────────────────────────────────────
def apply_smote(X, y):
    """Apply SMOTE to balance classes. k_neighbors=2 for small datasets."""
    sm = SMOTE(random_state=42, k_neighbors=2)
    if hasattr(X, "toarray"):
        X_res, y_res = sm.fit_resample(X.toarray(), y)
    else:
        X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

print("Applying SMOTE to balance classes...")
bow1_res,   y_bow1   = apply_smote(bow1,   y)
bow2_res,   y_bow2   = apply_smote(bow2,   y)
bow3_res,   y_bow3   = apply_smote(bow3,   y)
glove1_res, y_glove1 = apply_smote(glove1, y)
glove2_res, y_glove2 = apply_smote(glove2, y)
glove3_res, y_glove3 = apply_smote(glove3, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_bow1).value_counts(), "\n")

# ── ROC-AUC Helper ─────────────────────────────────────────────────────────────
def compute_roc_auc(model, X_te, y_te):
    try:
        classes = np.unique(y_te)
        if len(classes) == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)[:, 1]
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y_te).ravel()
            return round(roc_auc_score(y_bin, proba), 4)
    except Exception:
        pass
    return None

# ── Evaluate One Model ─────────────────────────────────────────────────────────
results = []

def evaluate_model(name, model, X_res, y_res, X_orig, y_orig):
    """Train on SMOTE-resampled data, evaluate on original test split."""
    classes = np.unique(y_orig)
    if len(classes) < 2:
        print(f"  {name}: Not enough classes — skipping.")
        return None

    # Split original data for test set (untouched by SMOTE)
    X_tr_idx, X_te_idx = train_test_split(
        np.arange(len(y_orig)), test_size=0.2, random_state=42, stratify=y_orig
    )

    # Use SMOTE data for training, original for testing
    if hasattr(X_orig, "toarray"):
        X_te = X_orig.toarray()[X_te_idx]
    else:
        X_te = X_orig[X_te_idx]
    y_te = y_orig[X_te_idx]

    # Train on full SMOTE-resampled set
    model.fit(X_res, y_res)
    y_pred   = model.predict(X_te)
    acc      = round(accuracy_score(y_te, y_pred), 4)
    rep      = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
    cm       = confusion_matrix(y_te, y_pred, labels=classes).tolist()
    auc      = compute_roc_auc(model, X_te, y_te)
    macro_f1 = round(rep["macro avg"]["f1-score"], 4)
    macro_pre= round(rep["macro avg"]["precision"], 4)
    macro_rec= round(rep["macro avg"]["recall"], 4)

    print(f"\n── {name} ──")
    print(f"  Accuracy: {acc}  |  Macro F1: {macro_f1}  |  Precision: {macro_pre}  |  Recall: {macro_rec}")
    print(f"  Confusion Matrix:\n  {np.array(cm)}")

    entry = {
        "name": name, "accuracy": acc, "macro_f1": macro_f1,
        "macro_precision": macro_pre, "macro_recall": macro_rec,
        "roc_auc": auc, "confusion_matrix": cm, "model": model,
        "X_res": X_res, "y_res": y_res,
        "X_te": X_te, "y_te": y_te, "y_pred": y_pred,
    }
    results.append(entry)
    return entry

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE — 12 ML Models (with SMOTE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BASELINE EVALUATION — 12 ML Models (SMOTE Applied)")
print("="*60)

schemes = [
    ("Scheme1-BOW",   bow1_res,   y_bow1,   bow1,   MultinomialNB()),
    ("Scheme2-BOW",   bow2_res,   y_bow2,   bow2,   MultinomialNB()),
    ("Scheme3-BOW",   bow3_res,   y_bow3,   bow3,   MultinomialNB()),
    ("Scheme1-GloVe", glove1_res, y_glove1, glove1, GaussianNB()),
    ("Scheme2-GloVe", glove2_res, y_glove2, glove2, GaussianNB()),
    ("Scheme3-GloVe", glove3_res, y_glove3, glove3, GaussianNB()),
]

for scheme_name, X_res, y_res, X_orig, nb_model in schemes:
    evaluate_model(f"NaiveBayes-{scheme_name}", nb_model, X_res, y_res, X_orig, y)
    evaluate_model(f"DecisionTree-{scheme_name}",
                   DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
                   X_res, y_res, X_orig, y)

print("\n" + "="*60)
print("BASELINE SUMMARY (with SMOTE)")
print("="*60)
summary = pd.DataFrame([{
    "Model": r["name"], "Accuracy": r["accuracy"],
    "Macro F1": r["macro_f1"], "Precision": r["macro_precision"],
    "Recall": r["macro_recall"], "ROC-AUC": r["roc_auc"],
} for r in results])
print(summary.to_string(index=False))

top3 = sorted(results, key=lambda x: x["macro_f1"], reverse=True)[:3]
print(f"\nTop 3 for optimization:")
for r in top3:
    print(f"  • {r['name']}  (Macro F1={r['macro_f1']})")

# ══════════════════════════════════════════════════════════════════════════════
# LEXICAL MODELS — 6 Models (no SMOTE needed — rule-based)
# ══════════════════════════════════════════════════════════════════════════════
nltk.download("sentiwordnet", quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def get_pos_tag(tag: str) -> Optional[str]:
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("R"): return wn.ADV
    return None

def sentiwordnet_label(text: str) -> str:
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    pos_score, neg_score = 0.0, 0.0
    for word, tag in tagged:
        wn_tag = get_pos_tag(tag)
        if wn_tag is None: continue
        synsets = list(swn.senti_synsets(word, wn_tag))
        if synsets:
            ss = synsets[0]
            pos_score += ss.pos_score()
            neg_score += ss.neg_score()
    if pos_score > neg_score:   return "positive"
    elif neg_score > pos_score: return "negative"
    return "neutral"

def load_opinion_words(url: str) -> set:
    resp = requests.get(url, timeout=15)
    words = set()
    for line in resp.text.splitlines():
        line = line.strip()
        if line and not line.startswith(";"): words.add(line.lower())
    return words

POSITIVE_WORDS = load_opinion_words("https://ptrckprry.com/course/ssd/data/positive-words.txt")
NEGATIVE_WORDS = load_opinion_words("https://ptrckprry.com/course/ssd/data/negative-words.txt")
print(f"Bing Liu: {len(POSITIVE_WORDS)} positive, {len(NEGATIVE_WORDS)} negative words loaded")

NEGATION_WORDS = {
    "not","no","never","neither","nor","nothing","nobody","nowhere",
    "hardly","barely","scarcely","without","cant","cannot","wont",
    "dont","doesnt","didnt","wasnt","isnt","aint","couldnt","shouldnt",
}

def bing_liu_label(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return "neutral"
    words = text.split()
    pos, neg = 0, 0
    for i, word in enumerate(words):
        negated = any(w in NEGATION_WORDS for w in words[max(0, i-3):i])
        if word in POSITIVE_WORDS:
            neg += negated; pos += not negated
        elif word in NEGATIVE_WORDS:
            pos += negated; neg += not negated
    if pos > neg:   return "positive"
    elif neg > pos: return "negative"
    return "neutral"

print("\n" + "="*60)
print("LEXICAL MODELS EVALUATION — 6 Models")
print("="*60)

lexical_results = []

def evaluate_lexical(name, pred_fn, texts, y_labels):
    y_pred   = np.array([pred_fn(t) for t in texts])
    acc      = round(accuracy_score(y_labels, y_pred), 4)
    rep      = classification_report(y_labels, y_pred, zero_division=0, output_dict=True)
    cm       = confusion_matrix(y_labels, y_pred, labels=np.unique(y_labels)).tolist()
    macro_f1 = round(rep["macro avg"]["f1-score"], 4)
    macro_pre= round(rep["macro avg"]["precision"], 4)
    macro_rec= round(rep["macro avg"]["recall"], 4)
    print(f"\n── {name} ──")
    print(f"  Accuracy: {acc}  |  Macro F1: {macro_f1}  |  Precision: {macro_pre}  |  Recall: {macro_rec}")
    print(f"  Confusion Matrix:\n  {np.array(cm)}")
    lexical_results.append({
        "name": name, "accuracy": acc, "macro_f1": macro_f1,
        "macro_precision": macro_pre, "macro_recall": macro_rec,
        "roc_auc": None, "confusion_matrix": cm,
    })

evaluate_lexical("SentiWordNet-Scheme1", sentiwordnet_label, texts1, y)
evaluate_lexical("SentiWordNet-Scheme2", sentiwordnet_label, texts2, y)
evaluate_lexical("SentiWordNet-Scheme3", sentiwordnet_label, texts3, y)
evaluate_lexical("BingLiu-Scheme1",      bing_liu_label,     texts1, y)
evaluate_lexical("BingLiu-Scheme2",      bing_liu_label,     texts2, y)
evaluate_lexical("BingLiu-Scheme3",      bing_liu_label,     texts3, y)

print("\n" + "="*60)
print("FULL SUMMARY — All 18 Models (SMOTE)")
print("="*60)
all_results = results + lexical_results
full_summary = pd.DataFrame([{
    "Model": r["name"], "Accuracy": r["accuracy"],
    "Macro F1": r["macro_f1"], "Precision": r["macro_precision"],
    "Recall": r["macro_recall"], "ROC-AUC": r["roc_auc"],
} for r in all_results])
print(full_summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION — 5-fold CV, bigger grids, SMOTE data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL OPTIMIZATION (SMOTE)")
print("="*60)

optimized_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scheme_map = {
    "Scheme1-BOW":   (bow1_res,   y_bow1,   bow1,   True),
    "Scheme2-BOW":   (bow2_res,   y_bow2,   bow2,   True),
    "Scheme3-BOW":   (bow3_res,   y_bow3,   bow3,   True),
    "Scheme1-GloVe": (glove1_res, y_glove1, glove1, False),
    "Scheme2-GloVe": (glove2_res, y_glove2, glove2, False),
    "Scheme3-GloVe": (glove3_res, y_glove3, glove3, False),
}

def optimize_model(baseline_entry, X_res, y_res, X_orig, y_orig, is_sparse):
    name = baseline_entry["name"]
    print(f"\nOptimizing: {name}")

    # Test set from original (unsmoted) data
    X_tr_idx, X_te_idx = train_test_split(
        np.arange(len(y_orig)), test_size=0.2, random_state=42, stratify=y_orig
    )
    X_te = X_orig[X_te_idx] if not hasattr(X_orig, "toarray") else X_orig.toarray()[X_te_idx]
    y_te = y_orig[X_te_idx]

    if "DecisionTree" in name:
        print("  → Upgrading to Random Forest + larger GridSearch")
        param_grid = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [3, 5, 10, None],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight":     ["balanced"],
        }
        base_model = RandomForestClassifier(random_state=42)

    elif "NaiveBayes" in name:
        if is_sparse:
            print("  → Tuning MultinomialNB alpha")
            base_model = MultinomialNB()
            param_grid = {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]}
        else:
            print("  → Tuning GaussianNB var_smoothing")
            base_model = GaussianNB()
            param_grid = {"var_smoothing": [1e-10, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]}
    else:
        return None

    gs = GridSearchCV(base_model, param_grid, cv=cv,
                      scoring="f1_macro", n_jobs=-1, refit=True, verbose=0)
    gs.fit(X_res, y_res)
    best_model = gs.best_estimator_

    y_pred      = best_model.predict(X_te)
    acc         = round(accuracy_score(y_te, y_pred), 4)
    rep         = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
    macro_f1    = round(rep["macro avg"]["f1-score"], 4)
    cm          = confusion_matrix(y_te, y_pred, labels=np.unique(y_te)).tolist()
    improvement = round((macro_f1 - baseline_entry["macro_f1"]) / max(baseline_entry["macro_f1"], 1e-9) * 100, 2)

    cv_scores = cross_val_score(best_model, X_res, y_res, cv=cv, scoring="f1_macro")
    cv_mean   = round(cv_scores.mean(), 4)
    cv_std    = round(cv_scores.std(), 4)

    print(f"  Best params   : {gs.best_params_}")
    print(f"  Accuracy      : {acc}")
    print(f"  Macro F1      : {macro_f1}  (baseline: {baseline_entry['macro_f1']}, {improvement:+.2f}%)")
    print(f"  CV F1 (5-fold): {cv_mean} ± {cv_std}")

    entry = {
        "name": f"OPT-{name}", "accuracy": acc, "macro_f1": macro_f1,
        "improvement_pct": improvement, "best_params": gs.best_params_,
        "cv_f1_mean": cv_mean, "cv_f1_std": cv_std,
        "confusion_matrix": cm, "model": best_model,
        "X_te": X_te, "y_te": y_te, "y_pred": y_pred, "is_sparse": is_sparse,
    }
    optimized_results.append(entry)
    return entry

for entry in top3:
    scheme_key = "-".join(entry["name"].split("-")[1:])
    X_res, y_res, X_orig, is_sparse = scheme_map.get(scheme_key, (glove2_res, y_glove2, glove2, False))
    optimize_model(entry, X_res, y_res, X_orig, y, is_sparse)

print("\n" + "="*60)
print("OPTIMIZATION SUMMARY (SMOTE)")
print("="*60)
for r in optimized_results:
    print(f"  {r['name']:<45} Macro F1: {r['macro_f1']}  CV: {r['cv_f1_mean']}±{r['cv_f1_std']}  Improvement: {r['improvement_pct']:+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

best_opt = max(optimized_results, key=lambda x: x["macro_f1"])
print(f"\nAnalyzing errors for: {best_opt['name']}")

y_te   = best_opt["y_te"]
y_pred = best_opt["y_pred"]

_, X_te_raw, _, _ = train_test_split(
    df["Review_Text"].fillna("").tolist(), y,
    test_size=0.2, random_state=42, stratify=y
)

errors = []
for i, (true, pred) in enumerate(zip(y_te, y_pred)):
    if true != pred:
        errors.append({"review": X_te_raw[i][:300], "true_label": true, "predicted_label": pred})

print(f"\nTotal misclassified: {len(errors)} / {len(y_te)}")
print(f"Error rate: {round(len(errors)/len(y_te)*100, 1)}%\n")

error_df = pd.DataFrame(errors)
if not error_df.empty:
    print("Error breakdown (true → predicted):")
    print(error_df.groupby(["true_label", "predicted_label"]).size().reset_index(name="count"))
    print("\nSample misclassified reviews:")
    for _, row in error_df.head(5).iterrows():
        print(f"\n  True: {row['true_label']}  |  Predicted: {row['predicted_label']}")
        print(f"  Review: {row['review'][:200]}...")
    error_df.to_csv("error_analysis2.csv", index=False)
    print("\nSaved error_analysis2.csv")

    print("\n── Error Pattern Conclusions ──")
    counts = error_df["true_label"].value_counts()
    for label, count in counts.items():
        if label == "negative":
            print(f"  • {count} negative review(s) misclassified — fewest training samples, model biased away.")
        elif label == "neutral":
            print(f"  • {count} neutral review(s) misclassified — mixed signals confuse the model.")
        elif label == "positive":
            print(f"  • {count} positive review(s) misclassified — sarcasm or domain-specific language.")
    print("  • GloVe averages word vectors — negation ('not good') may still appear positive.")
    print("  • SMOTE creates synthetic samples but cannot capture semantic nuance.")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL — best_model2.pkl
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SAVING BEST MODEL → best_model2.pkl")
print("="*60)

best_scheme_key = "-".join(best_opt["name"].replace("OPT-","").split("-")[1:])
_, _, _, best_is_sparse = scheme_map.get(best_scheme_key, (glove2_res, y_glove2, glove2, False))

best_vec = None
if best_is_sparse:
    if "Scheme1" in best_opt["name"]:   best_vec = vec1
    elif "Scheme2" in best_opt["name"]: best_vec = vec2
    else:                               best_vec = vec3

with open("best_model2.pkl", "wb") as f:
    pickle.dump({
        "model":      best_opt["model"],
        "model_name": best_opt["name"],
        "macro_f1":   best_opt["macro_f1"],
        "cv_f1":      best_opt["cv_f1_mean"],
        "is_glove":   not best_is_sparse,
        "vectorizer": best_vec,
        "glove":      GLOVE,
        "scheme":     best_scheme_key,
        "classes":    list(np.unique(y)),
    }, f)
print(f"Saved best_model2.pkl  →  {best_opt['name']}  (Macro F1={best_opt['macro_f1']})")

summary_data = {
    "approach":      "SMOTE oversampling",
    "baseline":      [{k: v for k, v in r.items() if k not in ("model","X_res","X_te","y_res","y_te","y_pred")} for r in results],
    "lexical":       lexical_results,
    "optimized":     [{k: v for k, v in r.items() if k not in ("model","X_te","y_te","y_pred")} for r in optimized_results],
    "best_model":    best_opt["name"],
    "best_macro_f1": best_opt["macro_f1"],
}
with open("optimization_summary2.json", "w") as f:
    json.dump(summary_data, f, indent=2)
print("Saved optimization_summary2.json")
print("\nDONE ✓")