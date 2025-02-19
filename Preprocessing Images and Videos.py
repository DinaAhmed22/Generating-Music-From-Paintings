import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Paths
input_dir = r"D:\Master\Selected Topics\labeled_images"
output_dir = r"D:\Master\Selected Topics\processed_images"
video_output_dir = os.path.join(output_dir, "video_frames")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Match CLIP/BLIP input size
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Function to apply noise reduction filters using OpenCV
def denoise_image_opencv(image):
    # Convert PIL image to NumPy array (OpenCV works with NumPy arrays)
    img_np = np.array(image)
    
    # Convert RGB to BGR as OpenCV uses BGR color order
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply Gaussian Blur to reduce noise
    img_denoised = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # Optionally, you can also apply Median or Bilateral Filtering
    # img_denoised = cv2.medianBlur(img_bgr, 5)
    # img_denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    # Convert back to RGB before returning as PIL image
    img_rgb = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    img_pil_denoised = Image.fromarray(img_rgb)
    return img_pil_denoised

# Function to enhance image contrast and sharpness
def enhance_image(image):
    # Convert image to NumPy array for OpenCV processing
    img_np = np.array(image)

    # Apply histogram equalization (for grayscale images) or CLAHE (for color images)
    if len(img_np.shape) == 3:  # For color images
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Equalize the Y channel (luminance)
        img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:  # For grayscale images
        img_np = cv2.equalizeHist(img_np)

    # Sharpen the image by applying a kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpen kernel
    img_sharpened = cv2.filter2D(img_np, -1, kernel)

    # Convert back to PIL Image
    img_pil_sharpened = Image.fromarray(img_sharpened)
    return img_pil_sharpened

### **Process Images (Ensure RGB Conversion and Denoising with OpenCV)**
def process_image(img_path, save_path):
    try:
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
        
        # Denoise image using OpenCV
        img = denoise_image_opencv(img)

        # Enhance image (contrast, sharpness)
        img = enhance_image(img)

        img = transform(img)  # Apply transformations

        # Convert back to PIL without forcing RGB
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(save_path, format="PNG")  # Save as PNG
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

### **Process Videos (Extract Frames & Preprocess)**
def process_video(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
    frame_interval = max(1, fps // frame_rate)  # Extract frames at specified FPS

    frame_count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert frame from BGR to RGB before processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL without color conversion
            
            # Denoise frame using OpenCV
            frame_pil = denoise_image_opencv(frame_pil)
            
            # Enhance frame (contrast, sharpness)
            frame_pil = enhance_image(frame_pil)

            frame_pil = transform(frame_pil)  # Apply transformations

            save_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame_{frame_id}.png")
            frame_pil_pil = transforms.ToPILImage()(frame_pil)
            frame_pil_pil.save(save_path, format="PNG")  # Save as PNG

            frame_id += 1  # Increment frame counter

        frame_count += 1

    cap.release()
    print(f"✅ Processed {frame_id} frames from {video_path}")

### **Main Processing Loop**
for file_name in tqdm(os.listdir(input_dir)):
    file_path = os.path.join(input_dir, file_name)
    
    # Process Images
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        save_path = os.path.join(output_dir, file_name.replace(".jpg", ".png"))
        process_image(file_path, save_path)

    # Process Videos
    elif file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
        video_frame_folder = os.path.join(video_output_dir, os.path.splitext(file_name)[0])
        os.makedirs(video_frame_folder, exist_ok=True)
        process_video(file_path, video_frame_folder, frame_rate=1)  # Extract frames at 1 FPS

print("🎥 ✅ All images & videos processed successfully!")
