import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNBinary(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNBinary, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the input to the first fully connected layer
        self.fc1_input_dim = 64 * 28 * 28  # Assuming input images are 224x224

        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_input_dim)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_model(self, train_loader, criterion, optimizer, device):
        self.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = self(images)  # Forward pass
            loss = criterion(outputs, labels.float())  # Compute the loss
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model parameters
            
            running_loss += loss.item() * images.size(0)
            
            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)
            
            # Binarize probabilities
            predicted = (probabilities > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
    
        print(f"Training Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
        return epoch_loss, epoch_accuracy
    
    def evaluate_model(self, val_loader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0

        y_pred = []
        y_true = []

        with torch.no_grad():  # Disable gradient calculation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels.float())  # Compute the loss
                
                running_loss += loss.item() * images.size(0)

                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs)
                
                # Binarize probabilities
                predicted = (probabilities > 0.5).float()
                
                # Collect predictions and true labels
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                
                # Calculate correct predictions
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = correct / total
        
        print(f"Validation Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
        return y_true, y_pred