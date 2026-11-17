import json
import logging
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from utils.model_utils import set_dropout

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SIZE = 10000
EPOCHS = 20
BATCH_SIZE = 128
STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}

logger = logging.getLogger("homework")


def setup_logging(name):
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / f"{name}.log", "w", "utf-8")],
    )


def set_seed(seed=42):
    torch.manual_seed(seed)


def load_dataset(name, train):
    source = datasets.MNIST if name == "mnist" else datasets.CIFAR10
    data = source(DATA_DIR, train=train, download=True)
    mean, std = STATS[name]
    x = (torch.as_tensor(data.data).float() / 255 - torch.tensor(mean)) / torch.tensor(std)
    return x.flatten(1), torch.as_tensor(data.targets)


def get_loaders(name, train_size=TRAIN_SIZE, batch_size=BATCH_SIZE):
    x_train, y_train = load_dataset(name, train=True)
    x_test, y_test = load_dataset(name, train=False)
    train_data = TensorDataset(x_train[:train_size], y_train[:train_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1000)
    return train_loader, test_loader, x_train.shape[1]

#Обучение

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = criterion(output, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (output.argmax(1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=0.001, weight_decay=0.0, dropout_schedule=None):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start = time.time()
    for epoch in range(epochs):
        if dropout_schedule:
            first, last = dropout_schedule
            set_dropout(model, first + (last - first) * epoch / max(epochs - 1, 1))
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        logger.info("  epoch %2d/%d  train acc %.4f  test acc %.4f", epoch + 1, epochs, train_acc, test_acc)
    history["time"] = time.time() - start
    return history


def summarize(history):
    return {
        "train_acc": history["train_acc"][-1],
        "test_acc": history["test_acc"][-1],
        "best_test_acc": max(history["test_acc"]),
        "gap": history["train_acc"][-1] - history["test_acc"][-1],
        "overfit_epoch": history["test_loss"].index(min(history["test_loss"])) + 1,
        "stability": statistics.pstdev(history["test_acc"][-5:]),
        "time_s": history["time"],
    }


def save_results(folder, name, histories, table):
    path = RESULTS_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    data = {"table": table.to_dict("records"), "histories": histories}
    (path / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    logger.info("%s\n%s\n", name, table.round(4).to_string(index=False))


def make_table(histories, **columns):
    table = pd.DataFrame([{"model": name, **summarize(h)} for name, h in histories.items()])
    for column, values in columns.items():
        table.insert(1, column, values)
    return table
