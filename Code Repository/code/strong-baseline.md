
## **Strong Baseline Classifier**

The `strong-baseline.py` script implements a strong baseline classifier for detecting AI-generated essays using logistic regression and various linguistic features.

### **How to Use the Script**

#### **Prerequisites**

- Python 3.x installed on your system.
- Required Python packages: (as listed in requirement.txt)
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `spacy`
  - `nltk`
  - `textstat`
  - `tqdm`
  - `joblib`
  - `pyspellchecker` 


#### **Script Overview**

- **Input Arguments:**
  - `--train`: Path to the training CSV file.
  - `--test`: Path to the test CSV file.
  - `--output`: Path to save the output predictions CSV file.

- **Functionality:**
  - Extracts linguistic features from the essays, such as readability scores, syntactic features, discourse markers, lexical richness, sentence length, and word length.
  - Trains a logistic regression model using the extracted features.
  - Makes predictions on the test data.
  - Saves the predictions to the specified output file.

#### **Running the Script**

1. **Navigate to the Project Directory:**

   Open a terminal and navigate to the directory containing the `strong-baseline.py` script.

2. **Run the Script with Required Arguments:**

   ```bash
   python strong-baseline.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output_predictions.csv>
   ```

   **Example:**

   ```bash
   python strong-baseline.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/strong_baseline_predictions.csv
   ```

   - **Explanation:**
     - `data/train_data.csv`: The training dataset containing labeled essays.
     - `data/test_data.csv`: The test dataset for which predictions are to be made.
     - `output/predictions/strong_baseline_predictions.csv`: The file where predictions will be saved.

3. **Sample Output:**

   ```
   Extracting features from training data...
   Extracting features from test data...
   Training the logistic regression model...
   Making predictions on test data...
   Predictions saved to output/predictions/strong_baseline_predictions.csv
   ```
