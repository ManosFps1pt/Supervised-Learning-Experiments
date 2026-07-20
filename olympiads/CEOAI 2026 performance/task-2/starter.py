import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Load data from CSV files
train_data = pd.read_csv("./train_data.csv")
test_data = pd.read_csv("./test_data.csv")

# map columns 'c' and 'mask' to lists of integers
to_list = lambda x: list(map(int, x.split(",")))
train_data["c"] = train_data["c"].apply(to_list)
train_data["mask"] = train_data["mask"].apply(to_list)
test_data["c"] = test_data["c"].apply(to_list)

print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples.")

# Load model and tokenizer
model_path = "./pythia-14m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()
print(f"Vocabulary size: {tokenizer.vocab_size}")

# View the first example in human readable format.
example = train_data.iloc[0]
a = np.array(example['c'])[np.array(example['mask']) == 0]
b = np.array(example['c'])[np.array(example['mask']) == 1]

print(f"id: {example['id']}, a length: {len(a)}, b length: {len(b)}, c length: {len(example['c'])}")
print(f"Sentence A (first 30 tokens): {tokenizer.decode(a[:30])}...")
print(f"Sentence B (first 30 tokens): {tokenizer.decode(b[:30])}...")
print(f"Interleaved (first 30 tokens): {tokenizer.decode(example['c'][:30])}...")

# TODO: Predict the masks for the examples in test set

def predict_mask(example):
    # TODO: placeholder
    return [0] * len(example['c']) 

def evaluate(data, predict_fn):
    """Evaluate predict_fn on labeled data using symmetric mask accuracy."""
    total_acc = 0.0
    for example in data:
        true_mask = np.array(example['mask'])
        pred_mask = np.array(predict_fn(example))
        if len(pred_mask) != len(true_mask):
            continue
        acc = np.mean(pred_mask == true_mask)
        total_acc += max(acc, 1.0 - acc)
    return total_acc / len(data) if data else None

# Example: evaluate on training data
# accuracy = evaluate(train_data.to_dict('records'), predict_mask)
# print(f"Train accuracy: {accuracy:.4f}")

# create a submission csv file

with open('submission.csv', 'w') as f:
    f.write('subtaskID,datapointID,answer\n')
    for _, example in test_data.iterrows():
        mask = predict_mask(example)
        answer = '"' + ','.join(str(m) for m in mask) + '"'
        f.write(f'"1","{example["id"]}",{answer}\n')

