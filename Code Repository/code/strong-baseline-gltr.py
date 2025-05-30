import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import argparse
from sklearn.metrics import f1_score

# Check for MPS (Metal Performance Shaders) device on macOS or fall back to CPU
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"Using device: {device}")

class GLTR:
    def __init__(self, model_name='gpt2', threshold=0.5):
        """
        Initialize the GLTR class.
        :param model_name: The name of the transformer model to use (default 'gpt2').
        :param threshold: Probability threshold for prediction (default 0.5).
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.threshold = threshold  # Threshold for prediction

    def tokenize(self, text):
        """
        Tokenize the input text.
        :param text: Text to tokenize.
        :return: Tensor of tokenized text.
        """
        # Truncate the text to the maximum allowable length for GPT-2 (usually 1024 tokens)
        max_length = self.tokenizer.model_max_length  # Get the max length of GPT-2
        tokens = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        return tokens

    def get_word_probabilities(self, tokens):
        """
        Get the probabilities of each token in the sequence.
        :param tokens: Tokenized text (Tensor).
        :return: List of probabilities for each token.
        """
        with torch.no_grad():
            outputs = self.model(tokens)
            logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities

    def analyze_text(self, text):
        """
        Analyze the input text to check for machine-generated patterns.
        :param text: The input text.
        :return: A list of probabilities for each word.
        """
        tokens = self.tokenize(text)
        probabilities = self.get_word_probabilities(tokens)

        # Calculate the probabilities for each token in the sequence
        prob_list = []
        for i in range(len(tokens[0]) - 1):  # Loop through each token except the last one
            prob = probabilities[0, i, tokens[0, i + 1]].item()  # Get the probability of the next token
            prob_list.append(prob)
        return prob_list

def tune_proportion_threshold(dev_data, gltr_model):
    """
    Tune the proportion threshold using the development dataset.
    :param dev_data: DataFrame containing development data with 'text' and 'label'.
    :param gltr_model: Instance of GLTR class.
    :return: Optimal proportion threshold.
    """
    best_threshold = 0
    best_f1 = 0

    # Try proportion thresholds from 0.2 to 0.7 in steps of 0.05
    #for threshold in [i / 100 for i in range(20, 70, 5)]:
    for threshold in [i / 100 for i in [40]]:
        print(f'Testing threshold {threshold}')
        predictions = []
        for text in dev_data['text']:
            prob_list = gltr_model.analyze_text(text)
            high_prob_tokens = sum(1 for p in prob_list if p > gltr_model.threshold)
            proportion_high_prob = high_prob_tokens / len(prob_list)
            
            # Predict based on the current threshold
            prediction = 1 if proportion_high_prob > threshold else 0
            predictions.append(prediction)
        
        # Compute F1 score instead of accuracy
        f1 = f1_score(dev_data['generated'], predictions)
        print(f'F1 score is {f1}')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Best threshold: {best_threshold}, Best F1 score: {best_f1}')

    return best_threshold

def main(test_file, dev_file, output_file):
    # Load test and dev data
    test_data = pd.read_csv(test_file)
    dev_data = pd.read_csv(dev_file)

    # Initialize GLTR model (GPT-2)
    gltr_model = GLTR()

    # Tune the proportion threshold using dev data
    optimal_threshold = tune_proportion_threshold(dev_data, gltr_model)

    # Perform GLTR analysis on test data with the tuned threshold
    y_pred = []
    for text in test_data['text']:
        prob_list = gltr_model.analyze_text(text)
        high_prob_tokens = sum(1 for p in prob_list if p > gltr_model.threshold)
        proportion_high_prob = high_prob_tokens / len(prob_list)

        # Use the tuned threshold for prediction
        prediction = 1 if proportion_high_prob > optimal_threshold else 0
        y_pred.append(prediction)

    # Save predictions to output file
    predictions = pd.DataFrame({'generated': y_pred})
    predictions.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GLTR-like Analysis of Text with Proportion Threshold Tuning.')
    parser.add_argument('--dev', required = True, help='Path to the dev CSV file for proportion threshold tuning.')
    parser.add_argument('--test', required = True, help='Path to the test CSV file.')
    parser.add_argument('--output', required = True, help='Path to save the output predictions CSV file.')
    args = parser.parse_args()
    main(args.test, args.dev, args.output)
