import pandas as pd
import numpy as np
import argparse
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import textstat
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from spellchecker import SpellChecker
from tqdm import tqdm
import random
import string

# For perplexity-based features
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Initialize tools
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

# Load GPT-2 model for perplexity scoring
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

def compute_perplexity(text):
    # Compute perplexity using GPT-2
    # Truncate text if too long to avoid memory issues
    encoded_input = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encoded_input.input_ids.to(device)
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    perplexity = np.exp(loss)
    return perplexity


def synonym_substitution(text, num_subst=1):
    # Simple synonym substitution using WordNet
    # We will attempt to replace 'num_subst' non-stopword tokens with a synonym
    words = word_tokenize(text)
    candidates = [w for w in words if w.isalpha() and w.lower() not in stop_words]
    if not candidates:
        return text
    for _ in range(num_subst):
        chosen_word = random.choice(candidates)
        syns = wordnet.synsets(chosen_word)
        if syns:
            lemmas = syns[0].lemmas()
            if len(lemmas) > 1:
                # pick a different lemma
                new_word = random.choice(lemmas).name().replace('_', ' ')
                # substitute in text
                words = [new_word if w == chosen_word else w for w in words]
    return " ".join(words)

def character_perturbation(text, prob=0.01):
    # Randomly insert, delete or swap characters in the text to simulate adversarial attacks
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() < prob:
            op = random.choice(['insert', 'delete', 'swap'])
            if op == 'insert':
                chars.insert(i, random.choice(string.ascii_letters))
                i += 1
            elif op == 'delete' and len(chars) > 1:
                del chars[i]
            elif op == 'swap' and i < len(chars)-1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
                i += 2
            else:
                i += 1
        else:
            i += 1
    return "".join(chars)

def augment_text(text):
    # Combine augmentations
    # Randomly choose whether to apply synonym substitution
    if random.random() < 0.3:
        text = synonym_substitution(text, num_subst=1)
    # Randomly apply character perturbation
    if random.random() < 0.3:
        text = character_perturbation(text, prob=0.01)
    return text

def extract_features(text):
    features = {}
    
    # Readability Metrics
    # Original features
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    # Additional readability metrics
    features['coleman_liau_index'] = textstat.coleman_liau_index(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)

    # Spelling Errors
    words = word_tokenize(text)
    misspelled = spell.unknown(words)
    features['spelling_errors'] = len(misspelled) / len(words) if words else 0

    # Syntactic Features (POS counts)
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    total_tokens = len(doc)
    for k, v in pos_counts.items():
        features[f'pos_{doc.vocab[k].text}'] = v / total_tokens if total_tokens > 0 else 0

    # Discourse Markers
    discourse_markers = ['however', 'therefore', 'moreover', 'consequently', 'furthermore', 'nevertheless']
    tokens = [w.lower() for w in words]
    discourse_count = sum(1 for w in tokens if w in discourse_markers)
    features['discourse_markers'] = discourse_count / len(tokens) if tokens else 0

    # Lexical Richness
    words_alpha = [word for word in tokens if word.isalpha() and word not in stop_words]
    features['type_token_ratio'] = len(set(words_alpha)) / len(words_alpha) if len(words_alpha) > 1 else 0

    # Sentence Length
    sentences = sent_tokenize(text)
    words_in_text = words
    features['avg_sentence_length'] = len(words_in_text) / len(sentences) if sentences else 0

    # Word Length
    word_lengths = [len(w) for w in words_alpha]
    features['avg_word_length'] = sum(word_lengths)/len(word_lengths) if word_lengths else 0

    # Token Frequency Features (top-k tokens)
    # Extract frequencies of top N most common tokens to capture distribution differences
    # Limit to alpha tokens
    freq_dist = nltk.FreqDist(words_alpha)
    # top 10 frequent tokens ratio
    most_common_10 = freq_dist.most_common(10)
    total_alpha = len(words_alpha)
    for i, (wrd, cnt) in enumerate(most_common_10):
        features[f'top_token_{i}'] = cnt/total_alpha if total_alpha > 0 else 0

    # Perplexity Feature
    try:
        pp = compute_perplexity(text)
    except:
        pp = 1000.0  # fallback if perplexity computation fails
    features['gpt2_perplexity'] = pp

    return features

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def main(train_file, test_file, output_file):
    # Load data
    print("Loading data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Basic checks
    if 'text' not in train_data.columns or 'generated' not in train_data.columns:
        raise ValueError("Train file must contain 'text' and 'generated' columns.")
    if 'text' not in test_data.columns:
        raise ValueError("Test file must contain 'text' column.")

    # Data Augmentation: Apply on a portion of the training data (e.g., 20%)
    print("Augmenting training data...")
    aug_mask = np.random.rand(len(train_data)) < 0.2
    train_data.loc[aug_mask, 'text'] = train_data.loc[aug_mask, 'text'].apply(augment_text)

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

    # Split a validation set from training data for hyperparameter tuning
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Pipeline and hyperparameter tuning for Logistic Regression
    print("Tuning Logistic Regression...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l2']
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=3)
    grid_search.fit(X_subtrain, y_subtrain)
    print("Best parameters for Logistic Regression:", grid_search.best_params_)

    best_lr = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_lr.predict(X_val)
    val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(y_val, y_val_pred)
    print("Validation Results (Logistic Regression):")
    print(f"Accuracy: {val_accuracy:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1: {val_f1:.2f}")

    # Ensemble with a Random Forest for improved robustness
    print("Training Random Forest and creating an ensemble...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_subtrain, y_subtrain)

    # Simple ensemble: average probabilities from LR and RF
    val_prob_lr = best_lr.predict_proba(X_val)[:,1]
    val_prob_rf = rf.predict_proba(X_val)[:,1]
    val_ensemble_prob = (val_prob_lr + val_prob_rf) / 2
    val_ensemble_pred = (val_ensemble_prob > 0.5).astype(int)

    ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1 = evaluate_model(y_val, val_ensemble_pred)
    print("Validation Results (Ensemble LR+RF):")
    print(f"Accuracy: {ensemble_accuracy:.2f}, Precision: {ensemble_precision:.2f}, Recall: {ensemble_recall:.2f}, F1: {ensemble_f1:.2f}")

    # Final model choice: use ensemble on full training data
    print("Refitting final models on full training data...")
    best_lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predict on test data
    print("Making predictions on test data...")
    y_test_prob_lr = best_lr.predict_proba(X_test)[:,1]
    y_test_prob_rf = rf.predict_proba(X_test)[:,1]
    y_test_prob_ensemble = (y_test_prob_lr + y_test_prob_rf) / 2
    y_test_pred = (y_test_prob_ensemble > 0.5).astype(int)

    # Save predictions
    predictions = pd.DataFrame({
        'generated': y_test_pred,
        'probability': y_test_prob_ensemble
    })
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Optionally save the models
    joblib.dump(best_lr, 'final_lr_model.pkl')
    joblib.dump(rf, 'final_rf_model.pkl')
    print("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extended Strong Baseline Classifier.')
    parser.add_argument('--train', required=True, help='Path to the training CSV file containing "text" and "generated" columns.')
    parser.add_argument('--test', required=True, help='Path to the test CSV file containing "text" column.')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file.')
    args = parser.parse_args()
    main(args.train, args.test, args.output)
