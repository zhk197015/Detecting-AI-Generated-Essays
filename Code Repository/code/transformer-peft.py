import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# Check for GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

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

def apply_lora(model, use_dora):
    lora_config = LoraConfig(
        task_type="SEQ_CLS",  # Sequence classification task
        use_dora=use_dora,    # True for DoRA, False for LoRA 
        r=8,                  # Low-rank dimension
        lora_alpha=32,        # Scaling factor
        lora_dropout=0.1,     # Dropout rate
        target_modules=["q_lin", "v_lin"]  # Target modules in DistilBERT
    )
    return get_peft_model(model, lora_config)

def main(train_file, test_file, output_file, use_dora):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Split train data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['text'], train_data['generated'], test_size=0.1, random_state=42
    )
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    # Load pre-trained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

    # Apply PEFT
    model = apply_lora(model, use_dora)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        dataloader_pin_memory=False
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    if use_dora == True:
        method_peft = 'DoRA'
    else:
        method_peft = 'LoRA'
    print(f"Training the model with {method_peft}...")
    trainer.train()

    # Save the model
    model.save_pretrained('efficient_baseline_model')
    tokenizer.save_pretrained('efficient_baseline_model')

    # Predict on the test data
    print("Making predictions on test data...")
    predictions = []
    for text in test_data['text']:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(prob[1] > 0.5)
            predictions.append(pred)

    # Save predictions
    output_df = pd.DataFrame({'generated': predictions})
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DistilBERT model with optional LoRA tuning.')
    parser.add_argument('--train', required=True, help='Path to the train CSV file.')
    parser.add_argument('--test', required=True, help='Path to the test CSV file.')
    parser.add_argument('--output', required=True, help='Path to save the output predictions CSV file.')
    parser.add_argument('--dora', required=True, type=bool, default=False, help='Enable DoRA for LoRA tuning.')
    args = parser.parse_args()

    main(args.train, args.test, args.output, args.dora)
