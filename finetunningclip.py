from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from datasets import Dataset
import os
from PIL import Image
from tqdm import tqdm
import torch
import re
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler

# Load CLIP model and processor
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the directory where your images are stored
image_dir = r"D:\Master\Selected Topics\processed_images"  # Adjust this path to where your images are located

# Load previously generated captions from the text file
caption_file = r"D:\Master\Selected Topics\processed_images\generated_captions.txt"
captions = {}

# Read captions from the file
with open(caption_file, "r") as f:
    for line in f:
        if line.strip():  # Avoid empty lines
            image_name, caption = line.split(":", 1)
            image_name = image_name.strip()  # Remove leading/trailing spaces
            caption = caption.strip()  # Remove leading/trailing spaces
            captions[image_name] = caption

# Get the full paths of the images in the directory
image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name in captions]
captions_list = [captions[os.path.basename(img_path)] for img_path in image_paths]  # List of corresponding captions

# Create a dictionary with image paths and captions for the dataset
data = {
    "image": image_paths,
    "caption": captions_list
}

# Create the dataset from the dictionary
dataset = Dataset.from_dict(data)

# Define a custom collate function
def collate_fn(batch):
    images = []
    captions = []
    
    for item in batch:
        img_path = item["image"]
        if os.path.exists(img_path):  # Check if image exists
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            captions.append(item["caption"])
        else:
            print(f"Image not found: {img_path}")

    # Process images and captions
    inputs = clip_processor(images=images, text=captions, padding=True, return_tensors="pt")
    
    return inputs

# Create a DataLoader
train_dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# Set up the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

# Training hyperparameters
num_epochs = 20  # Increased number of epochs for better convergence
initial_lr = 1e-5  # Lower initial learning rate to avoid instability
optimizer = Adam(clip_model.parameters(), lr=initial_lr)  # Adam optimizer with learning rate 1e-5
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)  # StepLR scheduler to reduce lr every 4 epochs by a factor of 0.1

# Start the training loop
for epoch in range(num_epochs):
    clip_model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        # Move batch to the device
        pixel_values = batch["pixel_values"].to(device)  # Image embeddings
        input_ids = batch["input_ids"].to(device)  # Text embeddings

        # Forward pass (no 'labels' argument)
        outputs = clip_model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Compute loss (contrastive loss between image and text embeddings)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Calculate contrastive loss (cross-entropy)
        labels = torch.arange(len(pixel_values), device=device)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels) + torch.nn.functional.cross_entropy(logits_per_text, labels)

        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)

        optimizer.step()  # Perform optimization step
        optimizer.zero_grad()  # Reset gradients

        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Optionally save the model, processor, and config after each epoch
    epoch_save_path = f"D:/Master/Selected Topics/Round3/clip/clip_finetuned_epoch_{epoch + 1}"
    clip_model.save_pretrained(epoch_save_path)
    clip_processor.save_pretrained(epoch_save_path)
    clip_model.config.save_pretrained(epoch_save_path)

    # Step the scheduler after each epoch
    scheduler.step()

# Save the final fine-tuned model, processor, and config
final_save_path = "D:/Master/Selected Topics/Round3/clip/clip_finetuned_final"
clip_model.save_pretrained(final_save_path)
clip_processor.save_pretrained(final_save_path)
clip_model.config.save_pretrained(final_save_path)
