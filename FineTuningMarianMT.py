import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import textwrap
import bidi.algorithm as bidi

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Load captions
caption_file = r"D:\Master\Selected Topics\processed_images\generated_captions.txt"
captions = []

with open(caption_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            _, caption = line.split(":", 1)
            captions.append(caption.strip())

# **Text Processing with Reshape and BiDi**
def preprocess_text(text):
    reshaped_text = textwrap.fill(text, width=60)
    rtl_text = bidi.get_display(reshaped_text)
    return rtl_text

# Apply preprocessing
captions_arabic = [preprocess_text(caption) for caption in captions]

# Create dataset
dataset = Dataset.from_dict({"source_text": captions, "target_text": captions_arabic})

# Tokenization function (fixes label padding issues)
def tokenize(batch):
    inputs = tokenizer(batch["source_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    targets = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Fix: Replace padding tokens with -100 to ignore loss on them
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0),
    }

# Tokenize dataset
dataset = dataset.map(tokenize, batched=True)

# Data collate function
def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
        "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]),
        "labels": torch.stack([torch.tensor(item["labels"]) for item in batch]),
    }

# DataLoader
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# **Fine-Tuning Setup**
num_epochs = 10
learning_rate = 2e-6  # Lower learning rate to stabilize training

optimizer = AdamW(model.parameters(), lr=learning_rate)

# **Save Training Configuration**
config = {
    "model_name": model_name,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "batch_size": 16,
    "max_length": 128,
    "device": str(device),
    "caption_file": caption_file,
}

# **Training Loop (Fix: Ignore <pad> tokens)**
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()

        # Clip gradients to prevent exploding gradient issue
        clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# **Save the fine-tuned model and config**
save_path = r"D:\Master\Selected Topics\MarianMT_Finetuned"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

config_path = os.path.join(save_path, "training_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Fine-tuned MarianMT model and training config saved at {save_path}")
