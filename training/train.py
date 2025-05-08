import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.model import TB_Detector
import os
from PIL import Image
import torch_directml
import time
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import StepLR

device = torch_directml.device()


normal_dir = "../data/TB_Chest_Radiography_Database/Normal"
tb_dir = "../data/TB_Chest_Radiography_Database/Tuberculosis"

class TB_Dataset(Dataset):
    def __init__(self, normal_dir, tb_dir, transform=None):
        self.transform = transform
        self.samples = []

        for label, dir_path in enumerate([normal_dir, tb_dir]):
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except:
            dummy_img = torch.zeros(3, 512, 512)
            return dummy_img, torch.tensor(0, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Change from 512x512
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DMLLoss(nn.Module):
    def forward(self, input, target):
        return torch.mean(-(target * torch.log(torch.sigmoid(input)) +
                            (1 - target) * torch.log(1 - torch.sigmoid(input))))


if __name__ == '__main__':

    dataset = TB_Dataset(normal_dir, tb_dir, transform)
    criterion = nn.BCELoss()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = TB_Detector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        model.train()
        start = time.time()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "../models/saved_models/best_tb_detector.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(val_loss)

    os.makedirs("../models/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "../models/saved_models/tb_detector.pt")
    print("Saved model to ../models/saved_models/tb_detector.pt")
