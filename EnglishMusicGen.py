# Import libraries
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import (
    pipeline, CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration,
    AutoProcessor, MusicgenForConditionalGeneration
)
from scipy.io.wavfile import write
import hashlib
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from scipy.linalg import sqrtm
from scipy.signal import spectrogram
from scipy.stats import entropy

# Input and output directories
input_dirs = r"D:\Master\Selected Topics\Round3\Image dataset-20250119T210031Z-001\Image dataset\v5"
output_dir = r"D:\Master\Selected Topics\Round3\output\musicop\op2"
os.makedirs(output_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")
# Initialize models
captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if torch.cuda.is_available() else -1)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

# Global variable for real-time processing
real_time_active = False
# Function to generate a caption using BLIP
def generate_caption(image):
    caption = captioning_pipeline(image)[0]["generated_text"]
    return caption

# Function to enhance caption using CLIP
def enhance_caption_with_clip(image, caption):
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        text_embeddings = outputs.text_embeddings
        image_embeddings = outputs.image_embeddings
    similarity = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings, dim=-1)
    similarity_score = similarity.item()
    if similarity_score > 0.8:
        enhanced_caption = f"{caption} (Highly relevant to the image)"
    else:
        enhanced_caption = f"{caption} (Moderately relevant to the image)"
    return enhanced_caption

# Function to enhance caption with musical context using T5
def enhance_caption_with_music_context(caption):
    prompt = f"""
    Enhance the following image description with musical context. Include details like mood, genre, tempo, and melody:
    Image Description: "{caption}"
    Musical Description:
    """
    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = t5_model.generate(**inputs, max_length=100)
    enhanced_caption = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced_caption

# Function to generate music from text using MusicGen
def generate_music_from_text(text_description, duration=10):
    inputs = musicgen_processor(text=[text_description], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_values = musicgen_model.generate(**inputs, max_new_tokens=int(duration * 50))
    return audio_values

# Function to generate a unique filename
def generate_unique_filename(caption, extension=".wav"):
    hash_object = hashlib.md5(caption.encode())
    hash_hex = hash_object.hexdigest()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{hash_hex}_{timestamp}{extension}"
    return filename

# Function to save and play the generated music
def save_and_play_music(audio_values, caption, sample_rate=32000, output_dir="output"):
    filename = generate_unique_filename(caption, extension=".wav")
    filepath = os.path.join(output_dir, filename)
    audio_array = audio_values.cpu().numpy().squeeze()
    write(filepath, sample_rate, audio_array)
    return filepath

# Function to save the image with captions
def save_image_with_captions(image, caption, enhanced_caption, output_dir="output"):
    filename = generate_unique_filename(enhanced_caption, extension=".png")
    filepath = os.path.join(output_dir, filename)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Caption: {caption}\nEnhanced Caption: {enhanced_caption}")
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    return filepath
# Function to calculate FAD (Frechet Audio Distance)
def calculate_fad_score(generated_audio, reference_audio):
    def extract_features(audio, sr):
        _, _, Sxx = spectrogram(audio, fs=sr)
        return np.mean(Sxx, axis=1)
    sr = musicgen_model.config.audio_encoder.sampling_rate
    gen_features = extract_features(generated_audio, sr)
    ref_features = extract_features(reference_audio, sr)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    mu_ref, sigma_ref = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen.dot(sigma_ref))
    fad = np.sum(diff**2) + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return fad

# Function to calculate CLAP score
def calculate_clap_score(text, audio):
    inputs = clip_processor(text=[text], audios=[audio], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        text_embeddings = outputs.text_embeddings
        audio_embeddings = outputs.audio_embeddings
    similarity = torch.nn.functional.cosine_similarity(text_embeddings, audio_embeddings, dim=-1)
    return similarity.item()

# Function to calculate THD (Total Harmonic Distortion)
def calculate_thd_score(audio):
    fft_vals = np.fft.fft(audio)
    fft_vals = np.abs(fft_vals)
    fundamental = np.max(fft_vals)
    harmonics = np.sum(fft_vals) - fundamental
    thd = harmonics / fundamental
    return thd

# Function to calculate KL Divergence
def calculate_kl_divergence(generated_audio, reference_audio):
    hist_gen, _ = np.histogram(generated_audio, bins=256, density=True)
    hist_ref, _ = np.histogram(reference_audio, bins=256, density=True)
    kl_divergence = entropy(hist_gen, hist_ref)
    return kl_divergence
# Function for real-time processing
def real_time_processing():
    global real_time_active
    real_time_active = True
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while real_time_active:
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = generate_caption(pil_image)
        enhanced_caption = enhance_caption_with_music_context(caption)
        audio_values = generate_music_from_text(enhanced_caption, duration=5)
        audio_filename = save_and_play_music(audio_values, enhanced_caption, output_dir=output_dir)
        image_filename = save_image_with_captions(pil_image, caption, enhanced_caption, output_dir=output_dir)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Tkinter GUI
def start_real_time_processing():
    global real_time_active
    if not real_time_active:
        threading.Thread(target=real_time_processing).start()
    else:
        messagebox.showinfo("Info", "Real-time processing is already active.")

def stop_real_time_processing():
    global real_time_active
    real_time_active = False
    messagebox.showinfo("Info", "Real-time processing stopped.")

def select_image():
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp")]
    )
    if image_path:
        image = Image.open(image_path)
        caption = generate_caption(image)
        enhanced_caption = enhance_caption_with_music_context(caption)
        audio_values = generate_music_from_text(enhanced_caption, duration=10)
        audio_filename = save_and_play_music(audio_values, enhanced_caption, output_dir=output_dir)
        image_filename = save_image_with_captions(image, caption, enhanced_caption, output_dir=output_dir)
        messagebox.showinfo("Info", f"Processing completed. Output saved in {output_dir}")

def main():
    root = tk.Tk()
    root.title("Music Generation from Images")
    select_image_button = tk.Button(root, text="Process Single Image", command=select_image)
    select_image_button.pack(pady=20)
    start_real_time_button = tk.Button(root, text="Start Real-Time Processing", command=start_real_time_processing)
    start_real_time_button.pack(pady=20)
    stop_real_time_button = tk.Button(root, text="Stop Real-Time Processing", command=stop_real_time_processing)
    stop_real_time_button.pack(pady=20)
    root.mainloop()
# Entry point
if __name__ == "__main__":
    main()