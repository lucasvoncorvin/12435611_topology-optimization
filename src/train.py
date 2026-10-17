from model import ViT
from dataset import TODataset, Mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import logging

N_EPOCHS = 10



def train(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn):
    model.train() # set to training mode
    
    n_samples = 0
    total_loss = 0
    
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # zero the gradients again so they dont sum up
        
        n_samples += x.shape[0]
        total_loss += loss.item() * x.shape[0]
        
        
    return total_loss / n_samples

def validate(model: torch.nn.Module, dataloader: DataLoader, loss_fn):
    model.eval() # set to evaluation mode
    
    n_samples = 0
    total_loss = 0
    
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        
        n_samples += x.shape[0]
        total_loss += loss.item() * x.shape[0]
        
    return total_loss / n_samples


def optimize():
    
    train_dataloader = DataLoader(
        dataset=TODataset(Mode.TRAIN),
        batch_size=256
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(Mode.VALIDATE),
        batch_size=256
    )
    loss_fn = nn.CrossEntropyLoss()
    vit = ViT()
    optimizer = optim.Adam(
        params=vit.parameters(),
        lr=0.001
    )
    
    for epoch in range(N_EPOCHS):
        
        train_loss = train(vit, train_dataloader, optimizer, loss_fn)
        test_loss = validate(vit, validate_dataloader, loss_fn)
        
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Test Loss {test_loss:.3f} | Delta {test_loss-train_loss:.3f}")
        
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG
    )
    optimize()
    