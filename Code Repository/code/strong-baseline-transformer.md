## **Transformer Baseline Classifier**

The `transformer-baseline.py` script implements a baseline classifier for detecting AI-generated text using a pre-trained BERT model fine-tuned for binary sequence classification.

### **How to Use the Script**

#### **Prerequisites**

- Python 3.x installed on your system.
- Required Python packages: (as listed in `requirements.txt`)
  - `pandas`
  - `argparse`
  - `torch`
  - `transformers`
  - `scikit-learn`

#### **Script Overview**

- **Input Arguments:**
  - `--train`: Path to the training CSV file.
  - `--test`: Path to the test CSV file.
  - `--output`: Path to save the output predictions CSV file.

- **Functionality:**
  - Loads and splits the training data into training and validation sets.
  - Prepares the data using BERT's tokenizer.
  - Fine-tunes a pre-trained BERT model for sequence classification.
  - Makes predictions on the test data.
  - Saves the predictions to the specified output file.

#### **Running the Script**

1. **Navigate to the Project Directory:**

   Open a terminal and navigate to the directory containing the `transformer-baseline.py` script.

2. **Run the Script with Required Arguments:**

   ```bash
   python transformer-baseline.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output_predictions.csv>

**Example:**

   ```bash
   python strong-baseline.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/strong_baseline_predictions.csv
   ```

3. **Sample Output**
```
Training the BERT model...
Making predictions on test data...
Predictions saved to output/predictions/strong_baseline_predictions.csv
```
