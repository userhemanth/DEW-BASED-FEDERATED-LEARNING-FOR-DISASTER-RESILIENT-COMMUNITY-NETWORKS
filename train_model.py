# src/train_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisasterCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(DisasterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # <-- Updated for 128x128 input
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 → 32
        x = self.pool(F.relu(self.conv3(x)))   # 32 → 16
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
