import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler

# Enable CUDA DSA for better debugging
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Load BLIP model and processor (using the large model for better performance)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

    # Process images and captions (tokenize captions as labels)
    inputs = processor(images=images, text=captions, padding=True, return_tensors="pt")
    
    # The processor returns a dictionary with input_ids and attention_mask,
    # but we also need to set the labels for training
    labels = inputs["input_ids"].clone()  # Use input_ids as labels for language modeling
    
    # Set the labels for training (mask padding tokens to -100 so they aren't used for loss computation)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Add labels to the inputs dictionary
    inputs["labels"] = labels
    
    return inputs

# Create a DataLoader
train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Set up the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training hyperparameters
num_epochs = 11 # Increased number of epochs for training
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-5)  # Keep the learning rate low for fine-tuning

# Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, verbose=True)


# Define the path for saving the model
save_path = r"D:\Master\Selected Topics\Round3\blip"

# Start the training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        # Move batch to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pixel_values = batch["pixel_values"].to(device)  # Ensure pixel_values are also passed

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        optimizer.step()  # Perform optimization step
        optimizer.zero_grad()  # Reset gradients

        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Save the model, processor, and config after each epoch
    epoch_save_path = os.path.join(save_path, f"blip_finetuned_epoch_{epoch + 1}")
    model.save_pretrained(epoch_save_path)
    processor.save_pretrained(epoch_save_path)
    model.config.save_pretrained(epoch_save_path)

    # Step the scheduler after each epoch
    scheduler.step(avg_loss)


# Save the final fine-tuned model, processor, and config
final_save_path = os.path.join(save_path, "blip_finetuned_final")
model.save_pretrained(final_save_path)
processor.save_pretrained(final_save_path)
model.config.save_pretrained(final_save_path)
