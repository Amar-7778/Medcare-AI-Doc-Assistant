import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

IMG_DIR = r"C:\Users\Amarnath\Downloads\archive (3)\sample\images"
CSV_PATH = r"C:\Users\Amarnath\Downloads\archive (3)\sample\sample_labels.csv"

MODEL_SAVE_NAME = "multi_disease_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class MultiDiseaseModel(nn.Module):
    def __init__(self):
        super(MultiDiseaseModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.tabular_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + 8, 64), nn.ReLU(),
            nn.Linear(64, 14), nn.Sigmoid()
        )

    def forward(self, img, tab):
        x_img = self.features(img).view(img.size(0), -1)
        x_tab = self.tabular_mlp(tab)
        return self.classifier(torch.cat((x_img, x_tab), dim=1))

class MultiLabelDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.scaler = StandardScaler()
        self.vitals = self.scaler.fit_transform(df[['Patient Age', 'Gender_Code', 'Temp_F', 'HR_bpm']].values)

    def __len__(self): return len(self.df)

    def get_labels(self, label_str):
        
        vec = np.zeros(14)
        for i, disease in enumerate(ALL_DISEASES):
            if disease in label_str: vec[i] = 1
        return vec

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.df.iloc[idx]['Image Index'])
        image = Image.open(path).convert('RGB')
        if self.transform: image = self.transform(image)
        vitals = torch.tensor(self.vitals[idx], dtype=torch.float32)
        label = torch.tensor(self.get_labels(self.df.iloc[idx]['Finding Labels']), dtype=torch.float32)
        return image, vitals, label

def train_and_evaluate():
    print(" Loading X-Ray Data...")
    df = pd.read_csv(CSV_PATH)

    def clean_age(age):
        s = str(age)
        if 'Y' in s: return int(s.replace('Y',''))
        if 'M' in s: return int(s.replace('M',''))
        if 'D' in s: return int(s.replace('D',''))
        return int(s)
    df['Patient Age'] = df['Patient Age'].apply(clean_age)

    np.random.seed(42)
    is_sick = df['Finding Labels'] != 'No Finding'
    df['Temp_F'] = np.where(is_sick, np.random.normal(100.5, 1.5, len(df)), np.random.normal(98.6, 0.5, len(df)))
    df['HR_bpm'] = np.where(is_sick, np.random.normal(100, 10, len(df)), np.random.normal(75, 8, len(df)))
    df['Gender_Code'] = df['Patient Gender'].apply(lambda x: 1 if x == 'M' else 0)

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    full_dataset = MultiLabelDataset(df, IMG_DIR, transform)

    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = MultiDiseaseModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() 
    
    print("\n Model Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for img, tab, lbl in train_loader:
            img, tab, lbl = img.to(DEVICE), tab.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(img, tab)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    print(f"\n Saving Model to {MODEL_SAVE_NAME}...")
    torch.save(model.state_dict(), MODEL_SAVE_NAME)

    print("\n Calculating Detailed Metrics...")
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img, tab, lbl in test_loader:
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            outputs = model(img, tab)
          
            preds = (outputs.cpu().numpy() > 0.3).astype(int)
            all_preds.append(preds)
            all_labels.append(lbl.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)

    print(f"\n===== FINAL RESULTS =====")
    print(f" Subset Accuracy: {subset_acc:.4f}")
    print(f" F1 Score (Micro): {f1:.4f}")
    
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=ALL_DISEASES, zero_division=0))

    print("\n CONFUSION MATRIX (Count of Correct Predictions per Disease):")
    print("-" * 50)
    header = f"{'DISEASE':<20} | {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5}"
    print(header)
    print("-" * 50)
    
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, disease in enumerate(ALL_DISEASES):
     
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"{disease:<20} | {tp:<5} {tn:<5} {fp:<5} {fn:<5}")
    print("-" * 50)

if __name__ == "__main__":
    train_and_evaluate()