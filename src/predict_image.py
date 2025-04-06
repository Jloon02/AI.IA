import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path

from model.cnn import VA_CNN  # Import the model architecture

# Configuration
IMAGE_SIZE = (128, 128)
MODEL_PATH = Path(__file__).parent / "trained_models" / "best_model.pth"  # Path to your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    """Load and preprocess a single image"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Apply the same transforms as during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict(image_path, model):
    """Predict valence and arousal for a single image"""
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Convert normalized predictions back to 1-10 scale
    valence_norm, arousal_norm = outputs[0].cpu().numpy()
    valence = int(round(valence_norm * 9 + 1))
    arousal = int(round(arousal_norm * 9 + 1))
    
    # Clip to ensure within bounds
    valence = max(1, min(10, valence))
    arousal = max(1, min(10, arousal))
    
    return valence, arousal

def main():
    # Load trained model
    model = VA_CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    # Path to your test image
    test_image_path = "neutral.jpg"  # Change this to your image path
    
    # Get prediction
    valence, arousal = predict(test_image_path, model)
    
    print(f"Predicted Valence (1-10): {valence}")
    print(f"Predicted Arousal (1-10): {arousal}")
    
    # Display the image with predictions
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (400, 400))  # Resize for display
    
    # Add text to image
    cv2.putText(img, f"Valence: {valence}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Arousal: {arousal}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()