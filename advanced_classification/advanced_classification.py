from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import torch
import time

"""What to do

1. Implement argparse to parse hyperparameters
2. Implement some kind of hyperparameter tuning
3. Implement model explainability
4. Add more evaluation metrics. Find all that can be used
5. Implement all with pytorch-lightning
6. Check for good practices to format code
7. Find for code optimalization

"""

ROOT = "./data"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = "cpu"


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
    print(f"Czas wykonania wynosi {total_time} sekund")
    
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

    print("Loading dataset...")
    train_dataset, test_dataset = create_datasets(root = ROOT)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader= DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print("Instantiating the model...")
    device = torch.device(DEVICE if torch.backends.mps.is_available() else "cpu")
    model = NeuralNet().to(device)
    #compiled_model = torch.compile(model, backend="cpu")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
                                 
    print("Initializing training...")
    train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)
   