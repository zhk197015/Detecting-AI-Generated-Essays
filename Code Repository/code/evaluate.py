# evaluate.py

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main(predictions_file, gold_file, output_dir):
    # Load predictions and gold labels
    preds = pd.read_csv(predictions_file)
    golds = pd.read_csv(gold_file)

    # Ensure the data is aligned
    assert len(preds) == len(golds), "Prediction and gold files must have the same length."
    assert 'generated' in preds.columns, "Predictions file must contain 'generated' column."
    assert 'generated' in golds.columns, "Gold file must contain 'generated' column."

    y_pred = preds['generated']
    y_true = golds['generated']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to a text file
    metrics_file_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write("===== Evaluation Metrics =====\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")

        f.write("===== Confusion Matrix =====\n")
        f.write(f"{cm_df}\n\n")

    # Output results
    print("===== Evaluation Metrics =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print confusion matrix to console
    print("\nConfusion Matrix:")
    print(cm_df)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # ===== Error Analysis =====
    # Merge predictions and gold labels for comparison
    results_df = golds.copy()
    results_df.reset_index(drop=True, inplace=True)
    results_df['predicted'] = y_pred.values

    # Identify misclassified samples
    misclassified = results_df[results_df['generated'] != results_df['predicted']]
    num_misclassified = len(misclassified)
    total_samples = len(results_df)
    misclassification_rate = num_misclassified / total_samples

    # Save error analysis to the text file
    with open(metrics_file_path, 'a') as f:
        f.write("===== Error Analysis =====\n")
        f.write(f"Total Misclassified Samples: {num_misclassified}/{total_samples} ({misclassification_rate:.2%})\n")

    print("\n===== Error Analysis =====")
    print(f"Total Misclassified Samples: {num_misclassified}/{total_samples} ({misclassification_rate:.2%})")

    # Save misclassified samples to a CSV file
    misclassified.to_csv(os.path.join(output_dir, 'misclassified_samples.csv'), index=False)
    print(f"Misclassified samples saved to {os.path.join(output_dir, 'misclassified_samples.csv')}")

    # Optional: Show examples of misclassified samples
    print("\nExamples of Misclassified Samples:")
    sample_errors = misclassified.head(3)  # Show first 3 misclassified samples
    with open(metrics_file_path, 'a') as f:
        f.write("\nExamples of Misclassified Samples:\n")
        for index, row in sample_errors.iterrows():
            example_text = f"\nSample {index + 1}:\nText: {row['text'][:200]}...\nActual Label: {row['generated']}\nPredicted Label: {row['predicted']}\n"
            print(example_text)
            f.write(example_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('--predictions', required=True, help='Path to the predictions CSV file.')
    parser.add_argument('--gold', required=True, help='Path to the gold standard CSV file.')
    parser.add_argument('--output_dir', required=True, help='Directory to save evaluation outputs.')
    args = parser.parse_args()
    main(args.predictions, args.gold, args.output_dir)
