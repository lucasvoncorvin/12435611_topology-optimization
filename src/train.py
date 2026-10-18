from model import ViT
from dataset import TODataset, Mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

@dataclass
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict
    epoch: int
    
@dataclass
class Hyperparameters:
    n_epochs: int
    embedding_dims: int
    query_dims: int
    n_attention_blocks: int
    optimizer: str
    learning_rate: float
    loss: str

def save_checkpoint(checkpoint: Checkpoint, filepath: Path):
    logging.info(f"Saving checkpoint (epoch {checkpoint.epoch}) to {filepath}")
    torch.save({
        "model_state_dict": checkpoint.model_state_dict,
        "optimizer_state_dict": checkpoint.optimizer_state_dict,
        "epoch": checkpoint.epoch
    }, filepath)
    
def load_checkpoint(filepath: Path) -> Checkpoint:
    # if model is checkpoint, load the checkpoint data
    logging.info(f"Loading checkpoint from {filepath}")
    checkpoint_data = torch.load(filepath)
    checkpoint = Checkpoint(
        model_state_dict=checkpoint_data["model_state_dict"],
        optimizer_state_dict=checkpoint_data["optimizer_state_dict"],
        epoch=checkpoint_data["epoch"]
    )
    return checkpoint

def make_optimizer(optimizer_name: str):
    """Optimizer factory
    
    returns the corresponding optimizer class based on the optimizer_name provided.
    """
    if optimizer_name == "adam":
        return optim.Adam
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def make_loss(loss_name: str):
    """Loss function factory
    
    returns the corresponding loss function class based on the loss_name provided.
    """
    if loss_name == "mse":
        return nn.MSELoss
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

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

def optimize(params: Hyperparameters):
    
    
    train_dataloader = DataLoader(
        dataset=TODataset(Mode.TRAIN),
        batch_size=256
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(Mode.VALIDATE),
        batch_size=256
    )
    loss_fn = make_loss(params.loss)()
    vit = ViT(
        embedding_dims=params.embedding_dims,
        query_dims=params.query_dims,
        n_attention_blocks=params.n_attention_blocks
    )
    logging.info(f"Initialized ViT model with embedding_dims={params.embedding_dims}, query_dims={params.query_dims}, n_attention_blocks={params.n_attention_blocks}, n_parameters={vit.n_weights}")
    
    
    optimizer = make_optimizer(params.optimizer)(
        params=vit.parameters(),
        lr=params.learning_rate
    )
    
    
    # set up model directory for saving checkpoints
    current_dir = Path.cwd()
    model_dir = current_dir.parent / "models"
    # create subdirectory for current model and checkpoints by using the model name
    model_subdir = model_dir / vit.name()
    training_subdir = model_subdir / f"{params.optimizer}_{params.learning_rate}_{params.loss}"
    training_subdir.mkdir(parents=True, exist_ok=True)
    
    
    # optimization loop
    for epoch in range(params.n_epochs):
        
        train_loss = train(vit, train_dataloader, optimizer, loss_fn)
        validation_loss = validate(vit, validate_dataloader, loss_fn)
        
        logger.info(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Validation Loss {validation_loss:.3f} | Delta {validation_loss-train_loss:.3f}")
        
        if epoch % 10 == 0:  # save checkpoint every 10 epochs
            checkpoint = Checkpoint(
                model_state_dict=vit.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch
            )
            
            save_checkpoint(checkpoint, training_subdir / f"epoch_{epoch}.pth") # todo: create naming convention for models and checkpoint
        
        
    # save model as final checkpoint
    final_model_path = training_subdir / f"epoch_{params.n_epochs}.pth"
    checkpoint = Checkpoint(
        model_state_dict=vit.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=params.n_epochs
    )
    save_checkpoint(checkpoint, final_model_path)
    
    return vit




if __name__ == "__main__":
    
    params = Hyperparameters(
        n_epochs=100,
        embedding_dims=32,
        query_dims=16,
        n_attention_blocks=4,
        optimizer="adam",
        learning_rate=0.001,
        loss="mse"
    )
    
    
    model = optimize(params)
    print("Training complete. Model:", model)
    
    # save the trained model
    torch.save(model.state_dict(), "vit_model.pth")