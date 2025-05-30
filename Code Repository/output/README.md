# Evaluation Metrics for AI-Generated Essay Detection

## Introduction

In this project, we aim to detect AI-generated essays using classification models. To evaluate the performance of these models, we use standard evaluation metrics for binary classification tasks. These metrics provide insights into various aspects of the model's performance, such as overall accuracy, precision in predicting positive cases, ability to recall actual positive cases, and the balance between precision and recall.

## Evaluation Metrics

### 1. Confusion Matrix

The confusion matrix is a table that describes the performance of a classification model by comparing the actual labels with the predicted labels. It is essential for understanding the types of errors the model is making.

**Structure of the Confusion Matrix:**

|                     | Predicted Negative (0) | Predicted Positive (1) |
|---------------------|------------------------|------------------------|
| **Actual Negative (0)** | True Negative (TN)       | False Positive (FP)      |
| **Actual Positive (1)** | False Negative (FN)      | True Positive (TP)       |

- **True Positive (TP):** Correctly predicted positive cases.
- **True Negative (TN):** Correctly predicted negative cases.
- **False Positive (FP):** Negative cases incorrectly predicted as positive.
- **False Negative (FN):** Positive cases incorrectly predicted as negative.

### 2. Accuracy

**Definition:**

Accuracy measures the proportion of total correct predictions out of all predictions made.

\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Explanation:**

- Reflects how often the model is correct overall.
- A higher accuracy indicates better overall performance.

### 3. Precision

**Definition:**

Precision measures the proportion of positive predictions that are actually correct.

\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{TP}{TP + FP}
\]

**Explanation:**

- High precision indicates that when the model predicts a sample as positive, it is likely correct.
- Crucial when the cost of false positives is high.

### 4. Recall (Sensitivity or True Positive Rate)

**Definition:**

Recall measures the proportion of actual positive cases that were correctly identified by the model.

\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{TP}{TP + FN}
\]

**Explanation:**

- High recall indicates that the model captures most of the actual positive cases.
- Important when missing positive cases is costly.

### 5. F1 Score

**Definition:**

The F1 score is the harmonic mean of precision and recall. It provides a balance between precision and recall.

\[
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**Explanation:**

- Useful when you need to find a balance between precision and recall.
- A higher F1 score indicates a better balance between the two.

## Calculation Example

Using the confusion matrix from your evaluation output:

|                     | Predicted 0 | Predicted 1 |
|---------------------|-------------|-------------|
| **Actual 0** (Negative) |   TN = 1689   |   FP = 83    |
| **Actual 1** (Positive) |   FN = 99    |   TP = 1044  |

### Calculations:

1. **Accuracy:**

   \[
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1044 + 1689}{1044 + 1689 + 83 + 99} = \frac{2733}{2915} \approx 0.9376
   \]

2. **Precision:**

   \[
   \text{Precision} = \frac{TP}{TP + FP} = \frac{1044}{1044 + 83} = \frac{1044}{1127} \approx 0.9264
   \]

3. **Recall:**

   \[
   \text{Recall} = \frac{TP}{TP + FN} = \frac{1044}{1044 + 99} = \frac{1044}{1143} \approx 0.9134
   \]

4. **F1 Score:**

   \[
   \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.9264 \times 0.9134}{0.9264 + 0.9134} \approx 0.9198
   \]

## References

- Wikipedia articles on [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix), [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall), and [F1 Score](https://en.wikipedia.org/wiki/F1_score).

## How to Run the Evaluation Script

To evaluate your model's predictions, use the following command:

```bash
python evaluate.py --predictions <path_to_predictions.csv> --gold <path_to_gold.csv> --output_dir <path_to_output_directory>
```

**Example:**

```bash
python evaluate.py --predictions output/predictions/strong_baseline_predictions.csv --gold data/test_data.csv --output_dir output/evaluation/strong_baseline
```

**Sample Output:**

```
===== Evaluation Metrics =====
Accuracy: 0.9376
Precision: 0.9264
Recall: 0.9134
F1 Score: 0.9198

Confusion Matrix:
           Predicted 0  Predicted 1
Actual 0         1689           83
Actual 1           99         1044

===== Error Analysis =====
Total Misclassified Samples: 182/2915 (6.24%)
Misclassified samples saved to output/evaluation/strong_baseline/misclassified_samples.csv

Examples of Misclassified Samples:

Sample 8:
Text: 
Many successful individuals have learned to cultivate resilience in the face of multiple failures. From Thomas Edison's over 1,000 failed attempts at inventing a working electric light bulb to J. K. ...
Actual Label: 1
Predicted Label: 0

Sample 30:
Text: Over the past few decades, talk regarding car pollution and usefulness has become common. Several people believe that the neglegence of cars is beneficial to the world, while others hold the belief th...
Actual Label: 0
Predicted Label: 1

Sample 84:
Text: Humans. One of the many smart species on planet earth. The species that seems to think they can do anything, including going to Venus. Although Venus is highly dangerous with its high temperatures and...
Actual Label: 0
Predicted Label: 1
```
