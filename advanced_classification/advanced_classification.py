from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import torch
import time
import argparse

"""What to do

1. Implement argparse to parse hyperparameters
2. Implement some kind of hyperparameter tuning
3. Implement model explainability
4. Add more evaluation metrics. Find all that can be used
5. Implement all with pytorch-lightning
6. Check for good practices to format code
7. Find for code optimalization

Next: Hydra config
"""

ROOT = "./data"


def create_datasets(root: str) -> None:
    train_dataset = datasets.MNIST(
        root = root,
        train = True,
        download = False,
        transform = ToTensor()
    )
    test_dataset = datasets.MNIST(
        root = root,
        train = False,
        download = False,
        transform = ToTensor()
    )
    return train_dataset, test_dataset

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")
   
def train(model, dataloader, loss_fn, optimizer, device, epochs):
    start_time = time.time()
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_epoch(model, dataloader, loss_fn, optimizer, device)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time is {total_time} seconds")
    
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs", type=int)
    parser.add_argument(
        "-lr", "--learning_rate", help="Learning rate", type=float)
    parser.add_argument(
        "-d", "--device", help="Device to use", type=str)
    parser.add_argument(
        "-b", "--batch_size", help="Batch size", type=int)
    parser.add_argument(
        "-r", "--root", help="Root directory", type=str, required=False)
    args = parser.parse_args()

    print("Loading dataset...")
    train_dataset, test_dataset = create_datasets(root = ROOT)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader= DataLoader(test_dataset, batch_size=args.batch_size)
    
    print("Instantiating the model...")
    device = torch.device(args.device)
    model = NeuralNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
                                 
    print("Initializing training...")
    train(model, train_dataloader, loss_fn, optimizer, args.device, args.epochs)
   