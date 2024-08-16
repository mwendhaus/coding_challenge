import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the input to the first fully connected layer
        self.fc1_input_dim = 64 * 28 * 28  # Assuming input images are 224x224

        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_input_dim)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_model(self, epochs, train_loader, criterion, optimizer, device):
        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()  # Zero the parameter gradients
                
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the model parameters
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            print(f"Training Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def evaluate_model(self, val_loader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                
                running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        
        print(f"Validation Loss: {epoch_loss:.4f}")