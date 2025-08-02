import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Add validation logic
    # Save best model (logic here)

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=class_names))
