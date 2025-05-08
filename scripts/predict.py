import torch
from torchvision import transforms
from PIL import Image
import os
from models.model import TB_Detector
import torch_directml

device = torch_directml.device()

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Taki sam jak w treningu
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = TB_Detector().to(device)
model.load_state_dict(torch.load("../models/saved_models/old_models/best_tb_detector.pt", map_location=device))
model.eval()

test_dir = "../data/test_images"

for img_name in os.listdir(test_dir):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            label = "Tuberculosis" if prob >= 0.7 else "Normal"
            print(f"{img_name}: {label} ({prob:.4f})")
