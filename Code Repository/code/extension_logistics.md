```markdown
# Extended Strong Baseline Classifier

This `extended_strong_baseline.py` script builds upon a simple baseline classifier for detecting AI-generated essays. It incorporates a variety of features (readability metrics, spelling errors, POS distributions, discourse markers, lexical richness, token frequency distributions, and GPT-2 perplexity scores), data augmentation techniques (synonym substitution and character perturbation), and an ensemble of models (Logistic Regression and Random Forest) to improve classification performance.

## How to Use the Script

### Prerequisites

- Python 3.x installed on your system.
- Required Python packages:
  - pandas
  - numpy
  - argparse
  - joblib
  - warnings
  - scikit-learn (for models and feature scaling)
  - textstat
  - spacy (with `en_core_web_sm` model installed)
  - nltk (with `stopwords` and `wordnet` downloaded)
  - pyspellchecker
  - tqdm
  - torch and transformers (for GPT-2 model)

**Installation:**
```bash
pip install pandas numpy argparse joblib scikit-learn textstat spacy nltk pyspellchecker tqdm torch transformers
```
  
**Download Spacy model:**
```bash
python -m spacy download en_core_web_sm
```

**Download NLTK resources:**  
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Script Overview

**Input Arguments:**

- `--train`: Path to the training CSV file containing "text" and "generated" columns.
- `--test`: Path to the test CSV file containing "text" column.
- `--output`: Path to save the output predictions CSV file.

**Functionality:**

1. **Data Loading**:  
   Loads the training and test datasets. Training data must have:
   - `text`
   - `generated`
   
   Test data must have:
   - `text`

2. **Data Augmentation**:
   - Random subset of training data is augmented via:
     - Synonym substitution using WordNet
     - Character-level perturbations (inserting, deleting, swapping characters)

3. **Feature Extraction**:
   - **Readability Metrics**: Flesch Reading Ease, Flesch-Kincaid Grade, Coleman-Liau Index, Automated Readability Index.
   - **Spelling Errors**: Ratio of misspelled words.
   - **POS Distributions**: Relative frequencies of different POS tags.
   - **Discourse Markers**: Frequency of words like "however", "therefore", etc.
   - **Lexical Richness**: Type-Token Ratio.
   - **Sentence and Word Length Metrics**: Average sentence length, average word length.
   - **Token Frequency Features**: Frequency of top 10 most common words.
   - **Perplexity Feature**: GPT-2 perplexity score.

4. **Model Training**:
   - Splits training data into sub-train and validation sets for hyperparameter tuning.
   - Uses `GridSearchCV` to tune a Logistic Regression model.
   - Trains a Random Forest classifier separately.
   - Combines both models (Logistic Regression and Random Forest) into an ensemble by averaging predicted probabilities.

5. **Evaluation**:
   - Evaluates on the validation set: accuracy, precision, recall, and F1-score.
   - Chooses the ensemble for the final model.

6. **Prediction on Test Data**:
   - Applies the final ensemble model on the test data.
   - Outputs predictions and probabilities to the specified output file.
   - Saves trained models for future use.

### Running the Script

**Navigate to the Project Directory:**
```bash
cd path/to/your/project
```

**Run the Script:**
```bash
python extended_strong_baseline.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output_predictions.csv>
```

**Example:**
```bash
python extended_strong_baseline.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/extended_strong_baseline_predictions.csv
```

- `data/train_data.csv`: Training dataset with `text` and `generated` columns.
- `data/test_data.csv`: Test dataset with `text` column.
- `output/predictions/extended_strong_baseline_predictions.csv`: Predictions output file.

### Sample Output

```
Loading data...
Augmenting training data...
Extracting features from training data...
Extracting features from test data...
Tuning Logistic Regression...
Best parameters for Logistic Regression: {'clf__C': 1, 'clf__penalty': 'l2'}
Validation Results (Logistic Regression):
Accuracy: 0.85, Precision: 0.84, Recall: 0.86, F1: 0.85
Training Random Forest and creating an ensemble...
Validation Results (Ensemble LR+RF):
Accuracy: 0.87, Precision: 0.86, Recall: 0.88, F1: 0.87
Refitting final models on full training data...
Making predictions on test data...
Predictions saved to output/predictions/extended_strong_baseline_predictions.csv
Models saved.
```

### Output File Format

The output CSV file contains:
- `generated`: The predicted class (0 = human-written, 1 = AI-generated).
- `probability`: The predicted probability of being AI-generated.

```