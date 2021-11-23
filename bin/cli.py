from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ser.data import train_data_loader
from ser.model import model1
from torchvision import datasets, transforms
from ser.model import model1
from ser.data import train_data_loader, val_data_loader
from ser.train import train_model
from ser.transforms import transform

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
    model, optimizer= model1(learning_rate, device)

    # torch transforms

    ts=transform()

    training_dataloader = train_data_loader(batch_size, ts, DATA_DIR)
    validation_dataloader= val_data_loader(batch_size, ts, DATA_DIR)

    train_model(epochs, training_dataloader, validation_dataloader, model, optimizer, device)
    # dataloaders

    # train



@main.command()
def infer():
    print("This is where the inference code will go")
