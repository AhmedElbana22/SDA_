import pandas as pd
import re
import string
import argparse
import nltk
from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Comprehensive text cleaning function"""
    if not isinstance(text, str):
        return ""
    
    # Remove markdown formatting & HTML tags 
    text = re.sub(r'\*\*.*?\*\*|_.*?_|\[.*?\]|<.*?>', '', text) 
    # Remove URLs 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions and hashtags 
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove non-ASCII characters 
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove repeated characters 
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip() 


def apply_lemmatization(text):
    """Apply lemmatization to text"""
    return " ".join([Word(word).lemmatize() for word in text.split()]) 

def remove_stopwords(text):
    """Remove stop words from text"""
    return " ".join([word for word in text.split() if word not in STOPWORDS])


def main():
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default="Movies.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="Final_Movies.csv", help="Output CSV file")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    col = 'Review_Text'

    df[col] = df[col].apply(clean_text)
    df[col] = df[col].apply(apply_lemmatization)
    df[col] = df[col].apply(remove_stopwords)

    df.to_csv(args.output, index=False) 

if __name__ == "__main__":
    main()