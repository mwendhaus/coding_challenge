import torch

def train_model(model, dataloader, criterion, optimizer, num_epochs=3, device='cpu'):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() 
            
            outputs = model(images).logits  
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step() 
            
            running_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)
        
            # Binarize probabilities
            predicted = (probabilities > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
        

def evaluate_model(model, val_loader, criterion, device="cpu"):
        model.eval() 
        running_loss = 0.0
        correct = 0
        total = 0

        y_pred = []
        y_true = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images).logits  
                loss = criterion(outputs, labels) 
                
                running_loss += loss.item() * images.size(0)
            
                probabilities = torch.sigmoid(outputs)
            
                # Binarize probabilities
                predicted = (probabilities > 0.5).float()
                
                # Collect predictions and true labels
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = correct / total
        
        print(f"Validation Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
        return y_true, y_pred