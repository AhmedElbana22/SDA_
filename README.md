# 🎬 Movie Sentiment Analysis — Social Data Analytics Project

A full sentiment analysis pipeline on movie reviews, covering ground truth labeling, text representation, lexical models, and machine learning classifiers.

---

## 📁 Project Structure

```
SDA_/
├── Movies.csv                  # Raw scraped movie reviews
├── Labeled_Movies.csv          # Reviews with ground truth labels
├── main.py                     # Text preprocessing functions
├── sentiment_pipeline.py       # Main pipeline script
├── bow_representation.json     # Bag-of-Words output
├── glove_representation.json   # GloVe output
├── requirements.txt            # Python dependencies
└── README.md
```

> ⚠️ `glove.6B.100d.txt` is **not included** due to its large size (331MB).  
> Download it manually from: https://nlp.stanford.edu/data/glove.6B.zip  
> Extract and place `glove.6B.100d.txt` in the project root folder.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AhmedElbana22/SDA_.git
cd SDA_
```

### 2. Create a Conda Environment

```bash
conda create -n SDA python=3.11
conda activate SDA
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

Run this once in Python:

```python
import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### 5. Set Your Groq API Key

Get a free API key from https://console.groq.com/keys, then set it as an environment variable:

USE it in `sentiment_pipeline.py` 

---

## 🔬 Pipeline Overview

| Step | Description |
|------|-------------|
| **Ground Truth Labeling** | 3 Groq (LLaMA) prompts per review → Fleiss Kappa inter-annotator agreement |
| **Text Representation** | Bag-of-Words + GloVe (3 preprocessing schemes each) |
| **Lexical Models** | SentiWordNet + Bing Liu dictionary with negation handling |
| **ML Models** | Naive Bayes + Decision Tree on all schemes |

---

## 📦 Requirements

See `requirements.txt`. Main dependencies:

- `groq` — LLM API for ground truth labeling
- `scikit-learn` — ML models
- `nltk` — Lexical models
- `pandas`, `numpy` — Data handling
- `requests` — Fetching Bing Liu dictionary
