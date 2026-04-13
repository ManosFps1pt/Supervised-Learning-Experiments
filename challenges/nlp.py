# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch

# 1. Load Data
dataset = load_dataset("imdb") 

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# 3. Load Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. Define Accuracy Metric (using the 'evaluate' library, NOT 'datasets.load_metric')
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Train + Evaluate
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    eval_strategy="epoch", 
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized["test"].select(range(500)),
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Predict on Custom Text
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

my_reviews = [
    "This movie was absolutely fantastic, I loved every second!",
    "What a terrible waste of time. Boring and predictable.",
    "It was okay, nothing special.",
]

print("\n--- Predictions ---")
for review, result in zip(my_reviews, classifier(my_reviews)):
    print(f"Review: {review}")
    print(f"  → {result['label']} (confidence: {result['score']:.2%})\n")

# 7. Save the Model
model.save_pretrained("./my_bert_sentiment")
tokenizer.save_pretrained("./my_bert_sentiment")
print("Model saved to ./my_bert_sentiment")
