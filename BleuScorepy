#####################Bleu_Score#################################################################################
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
#######################Code#####################################################################
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import BlipForConditionalGeneration, BlipProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Load the fine-tuned model
blip_model_path_finetuned = r"D:\Master\Selected Topics\Round3\blip\blip_finetuned_final"
model = BlipForConditionalGeneration.from_pretrained(blip_model_path_finetuned)
processor = BlipProcessor.from_pretrained(blip_model_path_finetuned)

input_dir = r"D:\Master\Selected Topics\processed_images"
reference_file_path = r"D:\Master\Selected Topics\processed_images\generated_captions.txt"
output_dir = r"D:\Master\Selected Topics\output_images"
os.makedirs(output_dir, exist_ok=True)

# Function to check if two images are identical
def are_images_identical(image1, image2):
    return np.array_equal(np.array(image1), np.array(image2))

# Function to read reference captions from .txt file
def load_reference_captions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [word_tokenize(line.strip().lower()) for line in file]

# Load reference captions
reference_captions = load_reference_captions(reference_file_path)

# Function to generate caption using fine-tuned BLIP model
def generate_finetuned_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Smoothing function for BLEU score
smoothing = SmoothingFunction()

# Initialize variables for duplicate detection
previous_image = None
image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))]

# Iterate over images and display them with their captions
for idx, file_name in enumerate(os.listdir(input_dir)):
    if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        if idx >= len(reference_captions):  # Prevent out-of-range error
            continue
        
        file_path = os.path.join(input_dir, file_name)
        caption = generate_finetuned_caption(file_path)
        generated_caption = word_tokenize(caption.lower())

        # Compute BLEU scores
        bleu_scores = [
            sentence_bleu([reference_captions[idx]], generated_caption, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1),
            sentence_bleu([reference_captions[idx]], generated_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method1),
            sentence_bleu([reference_captions[idx]], generated_caption, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing.method1),
            sentence_bleu([reference_captions[idx]], generated_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
        ]

        bleu_score = max(bleu_scores)

        # Skip printing low BLEU scores
        if bleu_score < 0.3:
            continue

        # Save image with caption
        img = Image.open(file_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = f"Caption: {caption}\nBLEU Score: {bleu_score:.4f}"
        draw.text((10, 10), text, font=font, fill="white")

        output_path = os.path.join(output_dir, f"output_{file_name}")
        img.save(output_path)

        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Generated Caption: {caption}\nBLEU Score: {bleu_score:.4f}")
        plt.show()
