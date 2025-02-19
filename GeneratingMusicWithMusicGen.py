import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os

# Set device to GPU or CPU depending on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model from Hugging Face
musicgen_model_path = "facebook/musicgen-small"  # Update with your model path if different
musicgen_processor = AutoProcessor.from_pretrained(musicgen_model_path)
musicgen_model = MusicgenForConditionalGeneration.from_pretrained(musicgen_model_path).to(device)

# Function to generate music from text using MusicGen
def generate_music_from_text(text_description, duration=10):
    # Preprocess the text description for MusicGen
    inputs = musicgen_processor(text=[text_description], padding=True, return_tensors="pt").to(device)
    
    # Generate the music (audio) for the current description
    with torch.no_grad():
        audio_values = musicgen_model.generate(**inputs, max_new_tokens=int(duration * 50))
        
    return audio_values

# Read the text file with descriptions
text_file_path = "D:\\Master\\Selected Topics\\T5_Music_Descriptions.txt"

# Read the text from the file
with open(text_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Set the directory path for saving generated music
output_directory = "D:\\Master\\Selected Topics\\Round3\\Tasleem\\Generated_Music_Fined_Tuned_T%"

# Make sure the directory exists, create it if necessary
os.makedirs(output_directory, exist_ok=True)

# Generate music for each description in the text file
for idx, line in enumerate(lines):
    line = line.strip()  # Clean up any extra whitespace
    
    if line:  # Ignore empty lines
        # Generate music from the description
        generated_audio = generate_music_from_text(line, duration=10)  # Adjust duration as needed
        
        # Convert the generated audio (tensor) to a format that can be saved
        audio_tensor = generated_audio[0].cpu()  # Ensure it's on the CPU for saving
        
        # Save the generated audio as a .wav file
        output_path = os.path.join(output_directory, f"generated_music_{idx + 1}.wav")
        torchaudio.save(output_path, audio_tensor, 22050)  # Save with a sample rate of 22050 Hz

        print(f"Generated music for description {idx + 1} and saved as '{output_path}'")
