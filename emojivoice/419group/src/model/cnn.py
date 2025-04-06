import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

from image_dataloader import VA_Dataset
from train import train_model

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
VAL_SPLIT = 0.2
TRAIN_DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "train"  # Path to your dataset folder
VAL_DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "test"  # Path to your dataset folder

class VA_CNN(nn.Module):
    def __init__(self):
        super(VA_CNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),  # Outputs both valence and arousal
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def main():
    # Define transforms
    transform = transforms.Compose([
        #transforms.Resize((48, 48)), # Resize to 48x48
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    train_dataset = VA_Dataset(TRAIN_DATA_PATH, transform=transform)
    val_dataset = VA_Dataset(VAL_DATA_PATH, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Initialize model
    model = VA_CNN()
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, val_loader, EPOCHS)
    print("Training completed!")

if __name__ == "__main__":
    main()