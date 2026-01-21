"""
Fine-tune a sentiment analysis model using Hugging Face Transformers
This script fine-tunes a DistilBERT model on your custom dataset
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import json
# Note: PyTorch is required but may have DLL issues on Windows
# Consider using Google Colab or cloud services for training

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Base model to fine-tune
OUTPUT_DIR = "./fine-tuned-model"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Example training data format:
# Each entry should have: {"text": "your text", "label": 0, 1, or 2}
# Labels: 0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE
TRAINING_DATA = [
    {"text": "I love this product!", "label": 2},
    {"text": "This is terrible.", "label": 0},
    {"text": "It's okay, nothing special.", "label": 1},
    {"text": "Absolutely amazing experience!", "label": 2},
    {"text": "Worst purchase I've ever made.", "label": 0},
    # Add more training examples here...
]

# Load tokenizer and model
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,  # NEGATIVE, NEUTRAL, POSITIVE
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
)

# Prepare dataset
def preprocess_function(examples):
    """Tokenize the texts"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512
    )

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(TRAINING_DATA)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split into train/validation (80/20 split)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)

# Compute metrics function
def compute_metrics(eval_pred):
    """Calculate accuracy"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("\n🚀 Starting training...")
trainer.train()

# Save the fine-tuned model
print(f"\n💾 Saving model to {OUTPUT_DIR}")
trainer.save_model()

# Evaluate
print("\n📊 Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

print("\n✅ Training complete!")
