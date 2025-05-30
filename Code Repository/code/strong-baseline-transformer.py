import pandas as pd
import argparse
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# Check for MPS (Metal Performance Shaders) device on macOS or fall back to CPU
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"Using device: {device}")

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze().to(device),
            'attention_mask': encoding['attention_mask'].squeeze().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }

def main(train_file, test_file, output_file):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare the training and testing datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['text'], train_data['generated'], test_size=0.1, random_state=42
    )
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
    test_dataset = TextDataset(test_data['text'].tolist(), [0] * len(test_data), tokenizer)  # Dummy labels

    # Load pre-trained BERT model for sequence classification and move it to the device
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs'
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    print("Training the BERT model...")
    trainer.train()

    # Save the model
    model.save_pretrained('strong_baseline_model')
    tokenizer.save_pretrained('strong_baseline_model')

    # Predict on the test data
    print("Making predictions on test data...")
    predictions = []
    for text in test_data['text']:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device

        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # Move to CPU for numpy
            pred = int(prob[1] > 0.5)
            predictions.append(pred)

    # Save predictions
    output_df = pd.DataFrame({'generated': predictions})
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strong Baseline Classifier using BERT.')
    parser.add_argument('--train', required=True, help='Path to the training CSV file.')
    parser.add_argument('--test', required=True, help='Path to the test CSV file.')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file.')
    args = parser.parse_args()
    main(args.train, args.test, args.output)
