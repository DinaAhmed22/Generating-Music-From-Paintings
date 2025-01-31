import os
import torch
import librosa
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    MusicgenForConditionalGeneration, MusicgenProcessor
)
from datasets import Dataset, Audio
from PIL import Image
from tqdm import tqdm
import soundfile as sf  # For saving audio files

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths to music datasets
music_dataset_paths = [
    r"D:\Master\Selected Topics\Round3\Omar Khairat Dataset\omar1",
    r"D:\Master\Selected Topics\musicsharky\ArabicClips\east",
    r"D:\Master\Selected Topics\musicsharky\ArabicClips\Muwa"
]

# Define path to image dataset
image_dir = r"D:/Master/Selected Topics/labeled_images"

# Define path to save preprocessed audio files
preprocessed_audio_dir = r"D:/Master/Selected Topics/preprocessed_audio"
os.makedirs(preprocessed_audio_dir, exist_ok=True)

# Ensure dataset paths exist
for path in music_dataset_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Music dataset path not found: {path}")
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")

# Load fine-tuned BLIP model and processor
blip_model_path = r"D:\Master\Selected Topics\Round3\Tasleem\fine_tuned_captioning_model"
blip_processor = BlipProcessor.from_pretrained(blip_model_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path).to(device)

# Load MusicGen model and processor
musicgen_processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

# Function to generate captions using the fine-tuned BLIP model
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = blip_model.generate(**inputs)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to load audio files from a directory
def load_audio_files(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Only process .wav files
                audio_files.append(os.path.join(root, file))
    return audio_files

# Function to preprocess and save audio files
def preprocess_and_save_audio(audio_files, target_sr=32000, target_length=10):
    preprocessed_files = []
    target_samples = target_length * target_sr  # Target length in samples

    for audio_file in tqdm(audio_files, desc="Preprocessing Audio"):
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=None)

        # Resample to target sampling rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Resize audio to target length
        if len(audio) < target_samples:
            # Pad with zeros if audio is shorter than target length
            padding = np.zeros(target_samples - len(audio))
            audio = np.concatenate([audio, padding])
        elif len(audio) > target_samples:
            # Truncate if audio is longer than target length
            audio = audio[:target_samples]

        # Save preprocessed audio file
        output_file = os.path.join(preprocessed_audio_dir, os.path.basename(audio_file))
        sf.write(output_file, audio, target_sr)
        preprocessed_files.append(output_file)

    return preprocessed_files

# Load all audio files from the music datasets
audio_files = []
for path in music_dataset_paths:
    audio_files.extend(load_audio_files(path))

# Preprocess and save audio files
preprocessed_audio_files = preprocess_and_save_audio(audio_files)

# Generate captions for the image dataset
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".jpeg", ".png"))]
captions = [generate_caption(image_path) for image_path in tqdm(image_paths, desc="Generating Captions")]

# Merge captions with the preprocessed audio files
# Ensure the number of captions matches the number of audio files
if len(captions) < len(preprocessed_audio_files):
    print(f"Warning: Only {len(captions)} captions available for {len(preprocessed_audio_files)} audio files. Truncating audio files.")
    preprocessed_audio_files = preprocessed_audio_files[:len(captions)]
elif len(captions) > len(preprocessed_audio_files):
    print(f"Warning: Only {len(preprocessed_audio_files)} audio files available for {len(captions)} captions. Truncating captions.")
    captions = captions[:len(preprocessed_audio_files)]

# Create a dataset from the preprocessed audio files and captions
merged_dataset = Dataset.from_dict({
    "audio": preprocessed_audio_files,
    "caption": captions
}).cast_column("audio", Audio())  # Cast the "audio" column to Audio type

# Preprocess the merged dataset for MusicGen
def preprocess_function(examples):
    # Process each audio file and caption individually
    audio_inputs = []
    text_inputs = []

    for audio, caption in zip(examples["audio"], examples["caption"]):
        # Preprocess audio
        audio_input = musicgen_processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True,  # Only padding is enabled
            truncation=False,  # Disable truncation
        )
        audio_inputs.append(audio_input["input_values"])  # Only use "input_values"

        # Preprocess caption
        text_input = musicgen_processor(
            text=caption,
            return_tensors="pt",
            padding="max_length",  # Pad to the maximum length in the batch
            truncation=False,  # Disable truncation
            max_length=512,  # Set a maximum length for padding
        )
        text_inputs.append(text_input["input_ids"])  # Only use "input_ids"

    # Stack inputs into batches
    return {
        "input_values": torch.cat(audio_inputs),  # Stack audio inputs
        "labels": torch.cat(text_inputs),  # Stack text inputs
    }

# Apply preprocessing to the merged dataset
encoded_dataset = merged_dataset.map(preprocess_function, batched=True, remove_columns=["audio", "caption"])

# Save the fine-tuned MusicGen model and processor
save_dir = "./fine_tuned_musicgen_model"
os.makedirs(save_dir, exist_ok=True)

musicgen_model.save_pretrained(save_dir)
musicgen_processor.save_pretrained(save_dir)

# Save model configuration
musicgen_model.config.to_json_file(os.path.join(save_dir, "config.json"))

print(f"Fine-tuned MusicGen model saved to: {save_dir}")