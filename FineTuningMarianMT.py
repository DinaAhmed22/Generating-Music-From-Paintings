from transformers import MarianMTModel, MarianTokenizer
import arabic_reshaper
from bidi.algorithm import get_display
import re

# Load MarianMT model and tokenizer for Arabic
model_name = "Helsinki-NLP/opus-mt-en-ar"  # Replace with your fine-tuned model if needed
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Path to the English captions file
caption_file = r"D:\Master\Selected Topics\processed_images\generated_captions.txt"

# Function to read English captions from the file
def read_english_captions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        captions = file.readlines()
    return [caption.strip() for caption in captions]  # Remove any trailing newlines

# Function to generate Arabic captions using MarianMT
def generate_arabic_captions(captions):
    arabic_captions = []
    for caption in captions:
        translated = model.generate(**tokenizer(caption, return_tensors="pt", padding=True))
        arabic_caption = tokenizer.decode(translated[0], skip_special_tokens=True)
        arabic_captions.append(arabic_caption)
    return arabic_captions

# Function to apply reshaping and BIDI processing
def process_arabic_text(arabic_text):
    # Reshape the Arabic text
    reshaped_text = arabic_reshaper.reshape(arabic_text)
    # Apply BIDI algorithm for correct text direction
    bidi_text = get_display(reshaped_text)
    # Clean text with regex (optional based on your need)
    cleaned_text = re.sub(r'[^\w\s]', '', bidi_text)  # Example: removing non-alphanumeric characters
    return cleaned_text

# Read English captions from the file
english_captions = read_english_captions(caption_file)

# Generate Arabic captions
arabic_captions = generate_arabic_captions(english_captions)

# Process each Arabic caption
processed_captions = [process_arabic_text(caption) for caption in arabic_captions]

# Path to save the Arabic captions
output_file = r"D:\Master\Selected Topics\processed_images\arabic_captions.txt"

# Save the processed Arabic captions to a text file
with open(output_file, "w", encoding="utf-8") as f:
    for caption in processed_captions:
        f.write(caption + "\n")

print(f"Arabic captions have been saved to {output_file}.")
