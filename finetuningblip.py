import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    MarianMTModel, MarianTokenizer, BlipProcessor, BlipForConditionalGeneration
)
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP model and processor for captioning
captioning_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(captioning_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(captioning_model_name).to(device)

# Load MarianMT model and tokenizer for translation
translation_model_name = "Helsinki-NLP/opus-mt-en-ar"
marian_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
marian_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

# Path to image dataset
image_dir = r"D:/Master/Selected Topics/labeled_images"

# Function to generate captions using BLIP
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to translate captions using MarianMT
def translate_caption(english_caption):
    inputs = marian_tokenizer(english_caption, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        translated = marian_model.generate(**inputs)
    
    translated_text = marian_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Function to create dataset
def create_dataset(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    data = {'image': [], 'caption': []}
    
    for image_path in tqdm(image_paths, desc="Generating Captions & Translating"):
        caption = generate_caption(image_path)
        arabic_caption = translate_caption(caption)
        
        data['image'].append(image_path)
        data['caption'].append(arabic_caption)
    
    return Dataset.from_dict(data)

# Create dataset
dataset = create_dataset(image_dir)

# Tokenize captions for MarianMT training
def preprocess_data(examples):
    inputs = marian_tokenizer(examples['caption'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = inputs['input_ids'].copy()  # Labels are the same as input_ids for MarianMT
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Create DataLoader with reduced batch size
dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)  # Reduced batch size

# Optimizers
optimizer_blip = AdamW(blip_model.parameters(), lr=1e-5)
optimizer_marian = AdamW(marian_model.parameters(), lr=5e-5)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Training for BLIP with gradient accumulation and mixed precision
gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
blip_model.train()
optimizer_blip.zero_grad()
for i, batch in enumerate(tqdm(dataloader, desc="Training BLIP")):
    # Process image batch for BLIP
    images = [Image.open(image_path).convert("RGB") for image_path in batch['image']]
    inputs_blip = blip_processor(images=images, return_tensors="pt").to(device)
    captions = batch['caption']
    text_inputs = blip_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Mixed precision training
    with autocast():
        outputs_blip = blip_model(pixel_values=inputs_blip.pixel_values, input_ids=text_inputs.input_ids, labels=text_inputs.input_ids)
        loss_blip = outputs_blip.loss / gradient_accumulation_steps  # Normalize loss
    
    scaler.scale(loss_blip).backward()  # Scale loss and perform backward pass
    
    # Perform optimizer step after accumulating gradients
    if (i + 1) % gradient_accumulation_steps == 0:
        scaler.step(optimizer_blip)
        scaler.update()
        optimizer_blip.zero_grad()

# Save fine-tuned models
save_dir_blip = "./fine_tuned_captioning_model"
save_dir_marian = "./fine_tuned_translation_model"

os.makedirs(save_dir_blip, exist_ok=True)
os.makedirs(save_dir_marian, exist_ok=True)

blip_model.save_pretrained(save_dir_blip)
blip_processor.save_pretrained(save_dir_blip)
marian_model.save_pretrained(save_dir_marian)
marian_tokenizer.save_pretrained(save_dir_marian)

print(f"Models saved:\n- BLIP: {save_dir_blip}\n- Marian: {save_dir_marian}")