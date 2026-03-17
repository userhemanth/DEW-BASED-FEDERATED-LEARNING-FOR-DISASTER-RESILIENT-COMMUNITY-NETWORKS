# src/client_device.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from train_model import DisasterCNN
import numpy as np
import random
import os
import time
import csv
from PIL import ImageFile

# Allow loading of truncated/corrupted images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== SETTINGS ==========
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 2
DATA_DIR = "data"  # main dataset folder

# Disaster classes
DISASTER_CLASSES = [
    "Drought",
    "Earthquake",
    "Human_Damage",
    "Infrastructure",
    "Land_Slide",
    "Urban_Fire",
    "Water_Disaster",
    "Wild_Fire",
    "Non_Damage"
]

# ========== LOCAL DATA LOADER ==========
def load_local_data(client_id):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    # Split dataset among 3 clients
    total_size = len(dataset)
    split_size = total_size // 3
    start = (client_id - 1) * split_size
    end = client_id * split_size if client_id < 3 else total_size

    local_subset = torch.utils.data.Subset(dataset, range(start, end))
    loader = torch.utils.data.DataLoader(local_subset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, dataset.classes

# ========== TRAIN & EVAL ==========
def train(model, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Client training - Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(trainloader):.4f}")

def evaluate(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

# ========== ALERT FUNCTION ==========
def trigger_disaster_alert(label):
    import time, os
    from notify import send_alert  # 🔔 import your notification helper

    alert_msg = f"[ALERT] {time.strftime('%H:%M:%S')} - Disaster detected: {label}"
    print(alert_msg)

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    # Log alert to file
    with open("data/local_alerts.log", "a") as f:
        f.write(alert_msg + "\n")

    # 🔔 Send a desktop notification
    try:
        send_alert(f"Disaster detected: {label}")
    except Exception as e:
        print(f"[WARN] Could not send desktop notification: {e}")

# ========== FLOWER CLIENT ==========
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, labels, client_id):
        self.model = model
        self.trainloader = trainloader
        self.labels = labels
        self.client_id = client_id
        os.makedirs("data", exist_ok=True)
        self.metrics_file = f"data/metrics_client_{client_id}.csv"
        # Write header if not exists
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "accuracy"])

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)

        # Random simulated detection for demo
        detected_label = random.choice(self.labels)
        if detected_label != "Non_Damage":
            trigger_disaster_alert(detected_label)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = evaluate(self.model, self.trainloader)
        print(f"Evaluation Accuracy: {acc:.2f}%")

        # Append to per-client CSV
        try:
            rnd = config.get("round", "")
        except Exception:
            rnd = ""
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rnd, acc])

        return float(0), len(self.trainloader.dataset), {"accuracy": acc}

# ========== MAIN ==========
if __name__ == "__main__":
    client_id = int(input("Enter client ID (1-3): "))
    model = DisasterCNN(num_classes=len(DISASTER_CLASSES))
    trainloader, labels = load_local_data(client_id)

    print(f"Client {client_id} connected with local dataset of {len(trainloader.dataset)} samples.")
    fl.client.start_numpy_client(
        server_address="localhost:9090",
        client=FlowerClient(model, trainloader, labels, client_id),
    )
