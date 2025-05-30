## **Simple Baseline Classifier**

The `simple-baseline.py` script implements a simple baseline classifier for detecting AI-generated essays. This baseline predicts the majority class found in the training data.

### **How to Use the Script**

#### **Prerequisites**

- Python 3.x installed on your system.
- Required Python packages: `pandas`, `numpy`, `argparse`.

#### **Script Overview**

- **Input Arguments:**
  - `--train`: Path to the training CSV file.
  - `--test`: Path to the test CSV file.
  - `--output`: Path to save the output predictions CSV file.

- **Functionality:**
  - Loads the training data to determine the majority class.
  - Predicts the majority class for all instances in the test data.
  - Saves the predictions to the specified output file.

#### **Running the Script**

1. **Navigate to the Project Directory:**

   Open a terminal and navigate to the directory containing the `simple-baseline.py` script.

2. **Run the Script with Required Arguments:**

   ```bash
   python simple-baseline.py --train <path_to_train.csv> --test <path_to_test.csv> --output <path_to_output_predictions.csv>
   ```

   **Example:**

   ```bash
   python simple-baseline.py --train data/train_data.csv --test data/test_data.csv --output output/predictions/simple_baseline_predictions.csv
   ```

   - **Explanation:**
     - `data/train_data.csv`: The training dataset containing labeled essays.
     - `data/test_data.csv`: The test dataset for which predictions are to be made.
     - `output/predictions/simple_baseline_predictions.csv`: The file where predictions will be saved.

3. **Sample Output:**

   ```
   The majority class is: 0
   Predictions saved to output/predictions/simple_baseline_predictions.csv
   ```

   - This output indicates that the majority class in the training data is `0`, and the predictions have been saved.

#### **Output File Format**

The output predictions CSV file will contain a single column:

- `generated`: The predicted class for each essay in the test data.