# LoRA-Enhanced DistilBERT Classifier

The script trains a DistilBERT-based classifier using **Low-Rank Adaptation (LoRA)** or **Weight-Decomposed LoRA (DoRA)** for efficient fine-tuning. It predicts whether a given text is generated or human-written.

---

## How to Use the Script

### Prerequisites
- Python 3.x installed on your system.
- Required Python packages:
  - `torch`
  - `transformers`
  - `peft`
  - `pandas`
  - `argparse`
  - `sklearn`

---

### Script Overview

#### **Input Arguments**
- `--train`: Path to the training CSV file.
- `--test`: Path to the test CSV file.
- `--output`: Path to save the output predictions CSV file.
- `--dora`: Boolean flag to enable **DoRA** for LoRA tuning (`True` or `False`). Use `--dora True` for DoRA. Default is `False`. 

#### **Functionality**
1. **Loads Training and Testing Data**:
   - Reads the training and testing datasets from CSV files.

2. **Tokenization**:
   - Uses Hugging Face's DistilBERT tokenizer to tokenize text data.

3. **LoRA/DoRA Fine-Tuning**:
   - Fine-tunes the DistilBERT model using **Parameter-Efficient Fine-Tuning (PEFT)**.
   - Supports both LoRA and DoRA depending on the `--dora` argument.

4. **Training and Validation**:
   - Splits the training data into training and validation sets.
   - Trains the model using Hugging Face's `Trainer` API.

5. **Prediction**:
   - Predicts whether texts in the test dataset are machine-generated.
   - Saves predictions to the specified output file.

---

### Running the Script

#### **1. Navigate to the Project Directory**
Open a terminal and navigate to the directory containing the script.

#### **2. Run the Script with Required Arguments**
```bash
python transformer-peft.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output.csv> --dora <True_or_False>
```

#### **Example**
```bash
python transformer-peft.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/transformer_peft_dora_predictions.csv --dora True
```

#### **Explanation**
- `data/train_data.csv`: The training dataset for fine-tuning the model.
- `data/test_data.csv`: The test dataset for predictions.
- `output/predictions/testing.csv`: The file where predictions will be saved.
- `--dora True`: Enables DoRA for LoRA tuning.

---

### Sample Output

1. **Training**:
```
Using device: cuda
Training the model with DoRA...
``` 

2. **Prediction**:
```
Making predictions on test data...
Predictions saved to output/predictions/transformer_peft_dora_predictions.csv
```

---

## Note
- If you want to run this on Colab T4 GPU to replicate our fast training time, use this Colab notebook: https://colab.research.google.com/drive/13V54rIAYT_vAzfJU_iUVQp4wh4yIUXUf?usp=sharing.
