import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import numpy as np
import logging
from scipy.signal import butter, filtfilt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_spectral_flatness(audio_tensor, sample_rate=22050, n_fft=1024, hop_length=512):
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True).abs() ** 2
    geometric_mean = torch.exp(torch.mean(torch.log(spectrogram + 1e-8), dim=-1))
    arithmetic_mean = torch.mean(spectrogram, dim=-1)
    return torch.mean(geometric_mean / (arithmetic_mean + 1e-8)).item()

def butter_lowpass_filter(data, cutoff=4000, fs=22050, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Set device to GPU or CPU depending on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for text descriptions
description_file = "D:\\Master\\Selected Topics\\T5_Music_Descriptions.txt"
output_directory = "D:\\Master\\Selected Topics\\Round3\\Tasleem\\Generated_Music"

# Load the processor and model
musicgen_model_path = "facebook/musicgen-small"
musicgen_processor = AutoProcessor.from_pretrained(musicgen_model_path)
musicgen_model = MusicgenForConditionalGeneration.from_pretrained(musicgen_model_path).to(device).to(torch.float32)

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Read descriptions from the text file
descriptions = []
try:
    with open(description_file, "r", encoding="utf-8") as file:
        descriptions = [line.strip() for line in file if line.strip()]
except Exception as e:
    logging.error(f"Error reading description file: {e}")
    exit(1)

# Function to generate music from text
def generate_music_from_text(text_description, duration=10):
    inputs = musicgen_processor(text=[text_description], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_values = musicgen_model.generate(inputs["input_ids"], max_new_tokens=int(duration * 50))
    return audio_values[0].cpu().numpy()  # Convert tensor to NumPy array

# Function to normalize audio

# Function to compute RMS Energy
def compute_rms_energy(audio_tensor):
    return torch.sqrt(torch.mean(audio_tensor ** 2)).item()


# Normalize audio tensor before computing ZCR
def normalize_audio(audio_tensor):
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val == 0:
        return audio_tensor  # Avoid division by zero
    return audio_tensor / (max_val + 1e-8)

# Function to compute Spectral Centroid
def compute_spectral_centroid(audio_tensor, sample_rate=22050, n_fft=1024, hop_length=512):
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True).abs()
    num_frequency_bins = spectrogram.shape[-2]  # Number of frequency bins
    frequencies = torch.linspace(0, sample_rate // 2, num_frequency_bins).to(spectrogram.device)  # Match frequency bins
    centroid = torch.sum(frequencies[:, None] * spectrogram, dim=-2) / (torch.sum(spectrogram, dim=-2) + 1e-8)
    return torch.mean(centroid).item()

# Function to compute Spectral Bandwidth
def compute_spectral_bandwidth(audio_tensor, sample_rate=22050, n_fft=1024, hop_length=512):
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True).abs()
    num_frequency_bins = spectrogram.shape[-2]  # Number of frequency bins
    frequencies = torch.linspace(0, sample_rate // 2, num_frequency_bins).to(spectrogram.device)  # Match frequency bins
    centroid = compute_spectral_centroid(audio_tensor, sample_rate, n_fft, hop_length)
    bandwidth = torch.sqrt(torch.sum((frequencies[:, None] - centroid) ** 2 * spectrogram, dim=-2) / (torch.sum(spectrogram, dim=-2) + 1e-8))
    return torch.mean(bandwidth).item()

# Process each text description
generated_metrics = {}
for idx, description in enumerate(descriptions):
    try:
        duration = max(5, len(description.split()) // 5)  # Dynamic duration based on description length
        generated_audio = generate_music_from_text(description, duration=duration)
        
        # Ensure generated_audio is correctly formatted
        if generated_audio.ndim == 3:
            generated_audio = generated_audio.squeeze(0)  # Remove batch dimension if present
        elif generated_audio.ndim == 1:
            generated_audio = generated_audio[None, :]  # Convert to 2D (mono)
        
        # Convert NumPy array to PyTorch tensor and normalize
        generated_audio_tensor = torch.tensor(generated_audio.copy(), dtype=torch.float32)
        generated_audio_tensor = normalize_audio(generated_audio_tensor)
        
        # Apply low-pass filter for noise reduction
        generated_audio_tensor = butter_lowpass_filter(generated_audio_tensor.numpy())
        generated_audio_tensor = torch.tensor(generated_audio_tensor.copy(), dtype=torch.float32)
        
        logging.info(f"Final tensor shape for description {idx}: {generated_audio_tensor.shape}")
        
        output_path = os.path.join(output_directory, f"generated_music_{idx}.wav")
        torchaudio.save(output_path, generated_audio_tensor, 22050)  # Ensure correct shape

        # Compute quality metrics
        spectral_flatness = compute_spectral_flatness(generated_audio_tensor)
        rms_energy = compute_rms_energy(generated_audio_tensor)
        spectral_centroid = compute_spectral_centroid(generated_audio_tensor)
        spectral_bandwidth = compute_spectral_bandwidth(generated_audio_tensor)
        
        generated_metrics[f"description_{idx}"] = {
            "Spectral Flatness": spectral_flatness,
            "RMS Energy": rms_energy,

            "Spectral Centroid": spectral_centroid,
            "Spectral Bandwidth": spectral_bandwidth,
        }
        
        logging.info(f"Generated music for description {idx} | Spectral Flatness: {spectral_flatness:.4f} | RMS Energy: {rms_energy:.4f}  | Spectral Centroid: {spectral_centroid:.4f} | Spectral Bandwidth: {spectral_bandwidth:.4f}")
    
    except Exception as e:
        logging.error(f"Error processing description {idx}: {e}")

# Save metrics to a text file
metrics_path = os.path.join(output_directory, "metrics_results.txt")
try:
    with open(metrics_path, "w") as metrics_file:
        for desc, scores in generated_metrics.items():
            metrics_file.write(f"{desc}: {scores}\n")
    logging.info(f"Metrics saved to {metrics_path}")
except Exception as e:
    logging.error(f"Error saving metrics: {e}")
