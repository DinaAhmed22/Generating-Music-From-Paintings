# Import libraries
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    pipeline, CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration,
    AutoProcessor, MusicgenForConditionalGeneration
)
from scipy.io.wavfile import write
import hashlib
from datetime import datetime
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from pygame import mixer  # For playing audio
import time

# Initialize pygame mixer for audio playback
mixer.init()

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

# Function to merge multiple images into one
def merge_images(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    merged_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return merged_image

# Function to process multiple images
def process_multiple_images(progress_bar):
    image_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp")]
    )
    if image_paths:
        try:
            progress_bar["value"] = 0
            progress_bar.update()
            images = [Image.open(image_path) for image_path in image_paths]
            progress_bar["value"] = 25
            progress_bar.update()
            merged_image = merge_images(images)
            progress_bar["value"] = 50
            progress_bar.update()
            caption = generate_caption(merged_image)
            enhanced_caption = enhance_caption_with_music_context(caption)
            progress_bar["value"] = 75
            progress_bar.update()
            audio_values = generate_music_from_text(enhanced_caption, duration=10)
            audio_filename = save_and_play_music(audio_values, enhanced_caption, output_dir=output_dir)
            image_filename = save_image_with_captions(merged_image, caption, enhanced_caption, output_dir=output_dir)
            progress_bar["value"] = 100
            progress_bar.update()
            messagebox.showinfo("Info", f"Processing completed. Output saved in {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            progress_bar["value"] = 0

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

# Function to process a video file
# Function to process a video file
def process_video(progress_bar):
    def open_file_dialog():
        video_path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if video_path:
            try:
                progress_bar["value"] = 0
                progress_bar.update()
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
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
                    progress_bar["value"] = (frame_count / total_frames) * 100
                    progress_bar.update()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Info", f"Processing completed. Output saved in {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                progress_bar["value"] = 0

    # Schedule the file dialog to run in the main thread
    app.after(0, open_file_dialog)
# Main Page
class MainPage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Page")
        self.geometry("600x400")

        # Title Label
        title_label = tk.Label(self, text="Music Generation from Images", font=("Arial", 24))
        title_label.pack(pady=20)

        # Button to navigate to the music generation page
        music_gen_button = tk.Button(self, text="Go to Music Generation", command=self.open_music_gen_page)
        music_gen_button.pack(pady=20)

    def open_music_gen_page(self):
        self.destroy()  # Close the main page
        MusicGenPage()  # Open the music generation page

# Music Generation Page
class MusicGenPage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Music Generation from Images and Videos")
        self.geometry("800x600")

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=20)

        # Buttons
        select_image_button = tk.Button(self, text="Process Single Image", command=self.process_single_image)
        select_image_button.pack(pady=20)

        select_multiple_images_button = tk.Button(self, text="Process Multiple Images", command=self.process_multiple_images)
        select_multiple_images_button.pack(pady=20)

        select_video_button = tk.Button(self, text="Process Video", command=self.process_video)
        select_video_button.pack(pady=20)

        start_real_time_button = tk.Button(self, text="Start Real-Time Processing", command=self.start_real_time_processing)
        start_real_time_button.pack(pady=20)

        stop_real_time_button = tk.Button(self, text="Stop Real-Time Processing", command=self.stop_real_time_processing)
        stop_real_time_button.pack(pady=20)

    def process_single_image(self):
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp")]
        )
        if image_path:
            try:
                self.progress_bar["value"] = 0
                self.progress_bar.update()
                image = Image.open(image_path)
                self.progress_bar["value"] = 25
                self.progress_bar.update()
                caption = generate_caption(image)
                enhanced_caption = enhance_caption_with_music_context(caption)
                self.progress_bar["value"] = 50
                self.progress_bar.update()
                audio_values = generate_music_from_text(enhanced_caption, duration=10)
                audio_filename = save_and_play_music(audio_values, enhanced_caption, output_dir=output_dir)
                image_filename = save_image_with_captions(image, caption, enhanced_caption, output_dir=output_dir)
                self.progress_bar["value"] = 100
                self.progress_bar.update()
                messagebox.showinfo("Info", f"Processing completed. Output saved in {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                self.progress_bar["value"] = 0

    def process_multiple_images(self):
        threading.Thread(target=process_multiple_images, args=(self.progress_bar,)).start()

    def process_video(self):
        threading.Thread(target=process_video, args=(self.progress_bar,)).start()

    def start_real_time_processing(self):
        global real_time_active
        if not real_time_active:
            threading.Thread(target=real_time_processing).start()
        else:
            messagebox.showinfo("Info", "Real-time processing is already active.")

    def stop_real_time_processing(self):
        global real_time_active
        real_time_active = False
        messagebox.showinfo("Info", "Real-time processing stopped.")

# Entry point
if __name__ == "__main__":
    app = MainPage()
    app.mainloop()