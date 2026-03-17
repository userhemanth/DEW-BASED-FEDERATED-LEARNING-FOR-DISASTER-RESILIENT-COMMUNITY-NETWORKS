# utils/data_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

    # Split training data among clients
    client_train_data = random_split(dataset, [int(len(dataset)/3)]*3)
    return client_train_data, testset

def get_data_loaders(trainset, testset, batch_size=32):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader
