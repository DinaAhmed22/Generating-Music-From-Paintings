mport torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torchvision
from torchvision import transforms

# Load BLIP model and processor (using the large model for better performance)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load Faster R-CNN model pre-trained on COCO
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()  # Set the model to evaluation mode

# Define a comprehensive list of Egyptian categories and famous museums
categories = {
    "Landmarks": [
        "pyramids", "Sphinx", "Nile River", "Karnak Temple", "Luxor Temple",
        "Valley of the Kings", "Abu Simbel", "Saqqara", "Alexandria Library","Fayoum","Cairo"
    ],
    "Artifacts": [
        "hieroglyphics", "ancient pottery", "jewelry", "sarcophagi", 
        "Tutankhamun's mask", "Rosetta Stone", "papyrus scrolls"
    ],
    "Art and Culture": [
        "calligraphy", "traditional clothing", "Nubian art", "Egyptian paintings", 
        "belly dancers", "traditional Egyptian music"
    ],
    "Famous Figures": [
        "Nefertiti", "Tutankhamun", "Cleopatra", "Ramses II", "King Farouk", 
        "Mohamed Ali Pasha", "Hatshepsut", "Imhotep","Pharaoh"
    ],
    "Museums": [
        "Egyptian Museum", "National Museum of Egyptian Civilization", 
        "Alexandria National Museum", "Giza Museum", "Military Museum", "Coptic Museum", 
        "Museum of Islamic Art", "Luxor Museum", "Grand Egyptian Museum"
    ],
    "Religion and Mythology": [
        "Ra", "Osiris", "Isis", "Horus", "Anubis", "Bastet", "Thoth", "Sekhmet", 
        "Amun", "Set", "Ma'at", "Nut", "Geb", "Ptah", "Tefnut", "Temples", "Sacred Texts",
        "Book of the Dead", "Pyramid Texts", "Coffin Texts", "Hieratic Texts", 
        "The Rosetta Stone"
    ],
    "Islamic Culture": [
        "Quran", "Islamic calligraphy", "mosque", "minaret", "Ramadan", "Eid", "Imam",
        "Muslim women", "hijab", "niqab", "Kaaba", "Makkah", "Medina", "Islamic prayer",
        "Salah", "Hajj", "Islamic architecture", "Cairo Mosque", "Al-Azhar Mosque",
        "Sufi", "Islamic scholars", "Fatimids", "Mamluks", "Islamic art"
    ],
    "Military Tools": [
        "tank", "missile", "rifle", "machine gun", "artillery", "military helicopter", 
        "ammunition", "grenade", "explosives", "bayonet", "army uniforms", "military boots", 
        "sword", "shield"
    ],
    "Military Museums": [
        "Egyptian Military Museum", "National Military Museum", "Cairo Military Museum", 
        "Mekka Military Museum", "Military Aviation Museum"
    ]
}

# Flatten categories to create a lookup table for automatic detection
all_keywords = {item: category for category, items in categories.items() for item in items}

# Mapping COCO class indices to both generic and Egyptian-related labels
object_labels = {
    1: "person", 2: "bicycle", 3: "car", 44: "bottle", 62: "chair", 63: "couch",
    67: "tv", 78: "keyboard", 80: "cell phone", 88: "laptop", 90: "toothbrush",
    1001: "Sphinx", 1002: "Great Pyramid of Giza", 1003: "Nile River", 1004: "Tutankhamun's Mask", 
    1005: "King Ramses II", 1006: "Cleopatra", 1007: "Luxor Temple", 1008: "Karnak Temple",
    1009: "Valley of the Kings", 1010: "Abu Simbel", 1011: "Egyptian Museum",
    1012: "National Museum of Egyptian Civilization", 1013: "Rosetta Stone", 
    1014: "Sarcophagus", 1015: "Hieroglyphics", 1016: "Papyrus Scrolls", 
    1017: "Coptic Museum", 1018: "Grand Egyptian Museum", 1019: "Luxor Museum", 
    1020: "Museum of Islamic Art", 1021: "Luxor Museum", 1022: "Saqqara",
    1023: "Karnak", 1024: "Giza Pyramids", 1025: "Seti I's Tomb",
    1026: "Ra", 1027: "Osiris", 1028: "Isis", 1029: "Horus", 1030: "Anubis", 
    1031: "Bastet", 1032: "Thoth", 1033: "Sekhmet", 1034: "Amun", 1035: "Set", 
    1036: "Ma'at", 1037: "Nut", 1038: "Geb", 1039: "Ptah", 1040: "Tefnut", 
    1041: "Quran", 1042: "Islamic Calligraphy", 1043: "mosque", 1044: "minaret", 
    1045: "Ramadan", 1046: "Eid", 1047: "Imam", 1048: "Muslim women", 1049: "hijab", 
    1050: "niqab", 1051: "Kaaba", 1052: "Makkah", 1053: "Medina", 1054: "Islamic Prayer",
    1055: "Salah", 1056: "Hajj", 1057: "Islamic Architecture", 1058: "Cairo Mosque", 
    1059: "Al-Azhar Mosque", 1060: "Sufi", 1061: "Islamic Scholars", 1062: "Fatimids", 
    1063: "Mamluks", 1064: "Islamic Art",
    1065: "Nefertiti", 1066: "Cleopatra", 1067: "Hatshepsut", 1068: "Imhotep", 1069: "Akhenaten", 
    1070: "Ramses II", 1071: "Thutmose III", 1072: "Amenhotep", 1073: "Seti I", 1074: "Merneptah",
    1075: "Obelisk", 1076: "Colossal Statues", 1077: "Hieratic Scroll", 1078: "Ancient Pottery", 
    1079: "Jewelry", 1080: "Mummy", 1081: "Tomb", 1082: "Papyrus", 1083: "Canopic Jar",
    1084: "Faiyum Portraits", 1085: "Theban Tombs", 1086: "Statue of Ramses II", 1087: "Cleopatra's Needle",
    1088: "Step Pyramid of Djoser", 1089: "Ptolemaic Architecture", 1090: "Egyptian Pyramids",
    1091: "Cairo Tower", 1092: "Alexandria Library", 1093: "Citadel of Saladin", 
    1094: "Tahrir Square", 1095: "Cairo Opera House", 1096: "Al-Azhar University", 
    1097: "Cairo University", 1098: "Nasser City", 1099: "Great Mosque of Muhammad Ali",
    1100: "Egyptian Parliament", 1101: "Nile Cruise", 1102: "El-Moez Street",
    1103: "The Book of the Dead", 1104: "Egyptian Calligraphy", 1105: "Nubian Art", 1106: "Coptic Art",
    1107: "Islamic Calligraphy", 1108: "Ancient Egyptian Art", 1109: "Pyramid Texts", 
    1110: "Egyptian Stelae", 1111: "Tutankhamun's Tomb", 1112: "Luxor Frescoes",
    1113: "Tank", 1114: "Missile", 1115: "Rifle", 1116: "Machine Gun", 1117: "Artillery", 
    1118: "Military Helicopter", 1119: "Ammunition", 1120: "Grenade", 1121: "Explosives", 
    1122: "Bayonet", 1123: "Army Uniforms", 1124: "Military Boots", 1125: "Sword", 1126: "Shield",
    1127: "Egyptian Military Museum", 1128: "National Military Museum", 
    1129: "Cairo Military Museum", 1130: "Mekka Military Museum", 1131: "Military Aviation Museum", 1132: "Egyptian Museum",1133:"Pharaoh",1134:"Fayoum",1135:"Cairo"
}

def get_specific_category(caption):
    """
    Determine specific category based on caption content
    """
    caption = caption.lower()
    
    for category, items in categories.items():
        if any(keyword in caption for keyword in items):
            return category
    return "General Egyptian Cultural Heritage"

def infer_category_from_object(object_label):
    """
    Map object label to one of the predefined Egyptian culture categories.
    """
    for category, items in categories.items():
        if object_label.lower() in items:
            return category
    return "General"  # Default category if no match is found

def generate_caption(image_path, category="General"):
    """
    Generate a caption for the image with a focus on the given category.
    """
    context = f"Egyptian culture, focusing on {category}"
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, text=context, return_tensors="pt")  # Context for generation
    out = model.generate(
        **inputs,
        max_length=50,  # Maximum length of the caption
        min_length=10,  # Minimum length of the caption
        num_beams=5,    # Beam search for better captions
        repetition_penalty=2.0  # Avoid repeated words
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.replace("focusing on ", "")  # Remove "focusing on " from the generated caption

def detect_objects_with_fasterrcnn(image_path):
    """
    Use Faster R-CNN to detect objects in an image.
    """
    transform = transforms.Compose([transforms.ToTensor()])  # Transformation to tensor
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Disable gradients to speed up inference
        prediction = faster_rcnn_model(img_tensor)  # Run Faster R-CNN

    # Get the labels and scores for objects detected with a confidence score
    detected_objects = []
    for element in range(len(prediction[0]["labels"])): 
        label = prediction[0]["labels"][element].item() 
        score = prediction[0]["scores"][element].item() 
        
        if score > 0.5:  # Only consider objects with a confidence score above 0.5
            detected_objects.append((label, score))
    
    # Map class indices to object labels (COCO class IDs and Egyptian-related objects)
    detected_objects = [(object_labels.get(label, "unknown"), score) for label, score in detected_objects]
    return detected_objects

def build_context(detected_objects):
    """Original context building function"""
    context = " Focusing on "
    categories_detected = set()
    
    for label, score in detected_objects:
        category = infer_category_from_object(label)
        categories_detected.add(category)
    
    context += ", ".join(categories_detected)
    return context

def generate_enhanced_caption(image_path):
    """
    Generate caption with specific category classification
    """
    # Get original caption
    base_caption = generate_caption(image_path)
    
    # Get specific category
    specific_cat = get_specific_category(base_caption)
    
    # Detect objects
    detected_objects = detect_objects_with_fasterrcnn(image_path)
    context = build_context(detected_objects)
    
    # Create enhanced caption
    enhanced_caption = f"{specific_cat} - {base_caption.replace('egyptian culture, ', '')}"
    if context.strip() != "Focusing on":
        enhanced_caption += f" ({context.replace('Focusing on ', '')})"
    
    return enhanced_caption

# Directory containing images
input_dir = r"D:\Master\Selected Topics\processed_images"
output_dir = r"D:\Master\Selected Topics\processed_images"
captions = {}

# Generate captions automatically based on filenames
for file_name in os.listdir(output_dir):
    if file_name.endswith(".png"):  # Process only image files
        file_path = os.path.join(output_dir, file_name)
        
        # Generate enhanced caption
        caption = generate_enhanced_caption(file_path)
        
        captions[file_name] = caption
        print(f"Generated caption for {file_name}: {caption}")

# Save captions to a file
with open(os.path.join(output_dir, "generated_captions.txt"), "w") as f:
    for img_name, caption in captions.items():
        f.write(f"{img_name}: {caption}\n")

print("Captions have been saved to generated_captions.txt.")
