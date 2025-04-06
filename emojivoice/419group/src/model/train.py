import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs, threshold=0.5):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            correct_train += torch.sum(torch.abs(outputs - labels) < threshold).item()
            total_train += labels.numel()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                correct_val += torch.sum(torch.abs(outputs - labels) < threshold).item()
                total_val += labels.numel()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'trained_models/best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Within {threshold}: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Within {threshold}: {val_acc:.2f}%')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    return model