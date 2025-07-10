
import test
import torch.utils.data.dataloader
from tqdm.auto import tqdm
import torch
import torch.nn as nn

def train_step(train_dataloader: torch.utils.data.DataLoader,
               model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device ,

               ):

        train_loss = 0
        train_acc = 0
        model.train()
        for X,y in train_dataloader:
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
            train_loss += loss
            train_acc += acc
        train_acc /= len(train_dataloader)
        train_loss /= len(train_dataloader)
        return train_loss, train_acc


def test_step(model:nn.Module,
              loss_fn:nn.Module,
              test_dataloader:torch.utils.data.DataLoader,
              device:torch.device):
    model.eval()
    with torch.inference_mode():

        test_loss = 0 
        test_acc = 0
        for X,y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            acc = (y_pred.argmax(dim=1)== y).sum().item() / len(y)
            test_loss += loss
            test_acc += acc
        test_acc/= len(test_dataloader)
        test_loss /= len(test_dataloader)
    return test_loss, test_acc


def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device:torch.device,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader):

    results = {
        "train_loss": [],
        "train_acc" : [],
        "test_loss" : [],
        "test_acc"  : []
    }
    for epoch in range(epochs):

        train_loss, train_acc =train_step(train_dataloader=train_dataloader,
                                            model= model,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss , test_acc = test_step(model= model,
                                         loss_fn=loss_fn,
                                         test_dataloader=test_dataloader,
                                         device=device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        print(f"EPOCH: {epoch+1}\n Train Loss : {train_loss} , Train Accuracy : {train_acc} | Test Loss : {test_loss} , Test Acc : {test_acc}")
    return results
