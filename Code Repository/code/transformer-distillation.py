import pandas as pd
import argparse
import os
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import logging
from tqdm import tqdm  # For progress bars

# Check for MPS (Metal Performance Shaders) device on macOS or fall back to CPU
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"Using device: {device}")

# Configure logging
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s %(message)s")
logging.info("Script started")

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

# Define distillation loss
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    ce_loss = F.cross_entropy(student_logits, labels)  # Cross-entropy with ground truth
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        soft_targets,
        reduction="batchmean"
    ) * (temperature ** 2)
    return alpha * kl_loss + (1 - alpha) * ce_loss

def main(train_file, test_file, output_file):
    # Load data
    logging.info("Loading data")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Initialize tokenizers
    teacher_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    student_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Prepare the datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data["text"], train_data["generated"], test_size=0.1, random_state=42
    )
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), teacher_tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), teacher_tokenizer)

    # Load teacher and student models
    logging.info("Loading teacher and student models")
    teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    # Set models to appropriate modes
    teacher_model.eval()
    student_model.train()

    # Define optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)

    # Training loop
    logging.info("Starting training loop")
    for epoch in range(3):  # Adjust the number of epochs
        student_model.train()
        total_loss = 0
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Get teacher logits
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Get student logits
            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_logits, labels)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Save the student model
    logging.info("Saving the student model")
    student_model.save_pretrained("distilled_model")
    student_tokenizer.save_pretrained("distilled_model")

    # Predict on the test data
    logging.info("Making predictions on test data")
    predictions = []
    for text in test_data["text"]:
        inputs = student_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = student_model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(prob[1] > 0.5)
            predictions.append(pred)

    # Save predictions
    output_df = pd.DataFrame({"generated": predictions})
    output_df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Text Classification.")
    parser.add_argument("--train", required=True, help="Path to the training CSV file.")
    parser.add_argument("--test", required=True, help="Path to the test CSV file.")
    parser.add_argument("--output", required=True, help="Path to save the output predictions CSV file.")
    args = parser.parse_args()
    main(args.train, args.test, args.output)
