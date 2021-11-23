from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ser1.data import train_data_loader
from ser1.model import model1
from torchvision import datasets, transforms
from ser1.model import model1
from ser1.data import train_data_loader, val_data_loader
from ser1.train import train_model
from ser1.transforms import transform

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int= typer.Option(2, "-eps", help="Number of epochs"),
    batch_size: int= typer.Option(1000, "-bs", "--batch_size", help="Batch size"),
    learning_rate: float= typer.Option(0.01, "-lr", "--learning rate", help= "learning rate")   
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """epochs = 2
    batch_size = 1000
    learning_rate = 0.01""" #replaced by typer options above 

    # save the parameters!

    # load model
    model, optimizer= model1(learning_rate)

    # torch transforms

    ts=transform()

    training_dataloader = train_data_loader(batch_size, ts, DATA_DIR)
    validation_dataloader= val_data_loader(batch_size, ts, DATA_DIR)

    train_model(epochs, training_dataloader, validation_dataloader, model, optimizer, device)
    # dataloaders

    # train


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output 


@main.command()
def infer():
    print("This is where the inference code will go")
