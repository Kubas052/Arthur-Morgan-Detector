# models/model.py
import torch.nn as nn
import torchvision.models as models

class TB_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # Użycie ResNet18 pretrenowanego
        self.model = models.resnet18(pretrained=True)
        # Zamiana ostatniej warstwy FC na 1 neuron (binarna klasyfikacja)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
