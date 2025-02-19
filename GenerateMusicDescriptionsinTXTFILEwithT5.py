import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load fine-tuned T5 model
t5_model_path = r"D:\Master\Selected Topics\T5_Finetuned"
tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata from the dataset
metadata_file = r"D:\Master\Selected Topics\processed_images\enhanced_captions_T5.txt"
generated_descriptions = []

with open(metadata_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            image_name, metadata = line.split(":", 1)  # Extract metadata
            input_text = f"Describe the music for: {metadata.strip()}"

            # Tokenize and generate description
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            outputs = model.generate(**inputs, max_length=128)
            description = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save generated description
            generated_descriptions.append(f"{image_name.strip()}|{description}")

# Save the descriptions as training data
output_file = r"D:\Master\Selected Topics\T5_Music_Descriptions.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in generated_descriptions:
        f.write(line + "\n")

print(f"Generated music descriptions saved to {output_file}")
