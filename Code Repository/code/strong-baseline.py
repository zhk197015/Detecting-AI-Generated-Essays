# strong-baseline.py 
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Feature extraction libraries
import textstat
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from tqdm import tqdm

# Initialize tools
nlp = spacy.load('en_core_web_sm')
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

def extract_features(text):
    features = {}
    
    # Readability Metrics
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    
    # Spelling Errors
    words = word_tokenize(text)
    misspelled = spell.unknown(words)
    features['spelling_errors'] = len(misspelled) / len(words) if words else 0
    
    # Syntactic Features
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    total_tokens = len(doc)
    for k, v in pos_counts.items():
        features[f'pos_{doc.vocab[k].text}'] = v / total_tokens  # Normalize by total tokens
    
    # Discourse Markers
    discourse_markers = ['however', 'therefore', 'moreover', 'consequently', 'furthermore', 'nevertheless']
    tokens = word_tokenize(text.lower())
    discourse_count = sum(1 for word in tokens if word in discourse_markers)
    features['discourse_markers'] = discourse_count / len(tokens) if tokens else 0
    
    # Lexical Richness
    words_alpha = [word for word in tokens if word.isalpha() and word not in stop_words]
    features['type_token_ratio'] = len(set(words_alpha)) / len(words_alpha) if words_alpha else 0
    
    # Sentence Length
    sentences = nltk.sent_tokenize(text)
    words_in_text = word_tokenize(text)
    features['avg_sentence_length'] = len(words_in_text) / len(sentences) if sentences else 0
    
    # Word Length
    word_lengths = [len(word) for word in words_alpha]
    features['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    return features

def main(train_file, test_file, output_file):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Extract features
    print("Extracting features from training data...")
    tqdm.pandas()
    X_train = train_data['text'].progress_apply(extract_features)
    X_train = pd.DataFrame(list(X_train))
    y_train = train_data['generated']

    print("Extracting features from test data...")
    X_test = test_data['text'].progress_apply(extract_features)
    X_test = pd.DataFrame(list(X_test))

    # Align train and test columns
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # Handle any remaining missing values
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Build a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Train the model
    print("Training the logistic regression model...")
    pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline, 'strong_baseline_model.pkl')

    # Predict on test data
    print("Making predictions on test data...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Save predictions
    predictions = pd.DataFrame({
        'generated': y_pred,
        'probability': y_prob
    })
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strong Baseline Classifier without Java dependency.')
    parser.add_argument('--train', required=True, help='Path to the training CSV file.')
    parser.add_argument('--test', required=True, help='Path to the test CSV file.')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file.')
    args = parser.parse_args()
    main(args.train, args.test, args.output)
