### **How to Use the Script**

#### **Prerequisites**

- Python 3.x installed on your system.
- Required Python packages: (as listed in `requirements.txt`)
  - pandas  
  - argparse  
  - torch  
  - transformers  
  - scikit-learn  
  - tqdm  

Install dependencies using:
```bash
pip install pandas argparse torch transformers scikit-learn tqdm
```

#### **Script Overview**

- **Input Arguments:**
  - `--train`: Path to the training CSV file (contains `text` and `generated` columns).
  - `--test`: Path to the test CSV file (contains `text` column).
  - `--output`: Path to save the output predictions CSV file.

- **Functionality:**
  - Loads and splits the training data into training and validation sets.
  - Prepares the data using DistilBERT's tokenizer with a maximum sequence length of 512 tokens.
  - Implements **knowledge distillation**:
    - Teacher Model: Pre-trained BERT (`bert-base-uncased`, frozen).
    - Student Model: DistilBERT (`distilbert-base-uncased`).
    - Loss: Combination of cross-entropy (hard labels) and KL divergence (soft labels).
  - Fine-tunes the DistilBERT model on the training set for 3 epochs.
  - Makes predictions on the test data using the distilled model.
  - Saves predictions to the specified output file in CSV format.

---

#### **Running the Script**

1. **Navigate to the Project Directory:**

   Open a terminal and navigate to the directory containing the `transformer_distillation.py` script.

2. **Run the Script with Required Arguments:**

```bash
python transformer_distillation.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output_predictions.csv>
```
**Example:**

```bash
python transformer_distillation.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/distilled_predictions.csv
```
3. **Sample Output**
Using device: cuda
Loading teacher and student models...
Starting training loop...
Epoch 1: 100%|█████████████████████████████| 1000/1000 [loss=0.1201]
Epoch 2: 100%|█████████████████████████████| 1000/1000 [loss=0.1023]
Epoch 3: 100%|█████████████████████████████| 1000/1000 [loss=0.0985]
Making predictions on test data...
Predictions saved to output/predictions/distilled_predictions.csv


