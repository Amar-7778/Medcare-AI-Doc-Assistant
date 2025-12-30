import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_DIR = r"C:\Users\Amarnath\Downloads\archive (5)\data\train" 

MODEL_SAVE_NAME = "skin_custom.pth"

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomMedicalCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomMedicalCNN, self).__init__()
        
  
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

def train_and_evaluate():
    print(f"Custom CNN Training on {DEVICE}...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    except Exception as e:
        print(f" Error: {e}")
        return

    classes = full_dataset.classes
    print(f" Detected {len(classes)} Classes: {classes}")

    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = CustomMedicalCNN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n Training Neural Network...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

 
    print(f"\n Saving Custom Architecture to {MODEL_SAVE_NAME}...")
    torch.save({
        'model_state': model.state_dict(),
        'classes': classes,
        'num_classes': len(classes)
    }, MODEL_SAVE_NAME)

    print("\n Calculating Metrics...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

  
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\n===== FINAL METRICS =====")
    print(f" Accuracy: {acc:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    print("\n CONFUSION MATRIX (Rows=True, Cols=Pred):")
    print("-" * 30)

    header = "      " + "  ".join([f"{c[:4]:>4}" for c in classes])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = "  ".join([f"{val:>4}" for val in row])
        print(f"{classes[i][:4]:>4} | {row_str}")
    print("-" * 30)

if __name__ == "__main__":
    train_and_evaluate()