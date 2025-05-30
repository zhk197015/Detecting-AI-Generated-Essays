# simple-baseline.py

import pandas as pd
import argparse
import os

def main(train_file, test_file, output_file):
    # Load training data to find the majority class
    train_data = pd.read_csv(train_file)
    majority_class = train_data['generated'].value_counts().idxmax()
    print(f"The majority class is: {majority_class}")

    # Load test data
    test_data = pd.read_csv(test_file)

    # Assign the majority class to all instances
    predictions = pd.DataFrame({'generated': [majority_class] * len(test_data)})

    # Save the predictions
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Majority Class Baseline.')
    parser.add_argument('--train', required=True, help='Path to the training CSV file.')
    parser.add_argument('--test', required=True, help='Path to the test CSV file.')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file.')
    args = parser.parse_args()
    main(args.train, args.test, args.output)
