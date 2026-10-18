from model import ViT
from dataset import TODataset, Mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
import logging
import dataclasses
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Hyperparameters:
    n_epochs: int
    embedding_dims: int
    query_dims: int
    n_attention_blocks: int
    n_attention_heads: int
    optimizer: str
    learning_rate: float
    loss: str
    # batch_size ?
    # n_training_samples ?
    
@dataclass
class TrainingResults:
    train_losses: list
    validation_losses: list
@dataclass
class Checkpoint:
    path: Path | None  # path to the checkpoint file, None if starting from scratch
    model_state_dict: dict | None  # none if starting from scratch
    optimizer_state_dict: dict | None 
    params: Hyperparameters
    results: TrainingResults | None

def plot_training_history(train_losses, validation_losses, path=None, show=True):
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(epochs, train_losses, label="Train Loss")
    plt.semilogy(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()

def save_checkpoint(checkpoint: Checkpoint, plot: bool = False):
    if checkpoint.path is None:
        raise ValueError("Checkpoint path is None. Cannot save checkpoint.")
    if checkpoint.results is None:
        checkpoint.results = TrainingResults(train_losses=[], validation_losses=[])
        logger.warning("Checkpoint results are None.")
    checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving checkpoint with hyperparameters {checkpoint.params} to {checkpoint.path}")
    torch.save({
        "model_state_dict": checkpoint.model_state_dict,
        "optimizer_state_dict": checkpoint.optimizer_state_dict,
        "hyperparameters": dataclasses.asdict(checkpoint.params),
        "results": dataclasses.asdict(checkpoint.results),
    }, checkpoint.path)
    if plot:
        plot_training_history(checkpoint.results.train_losses, checkpoint.results.validation_losses, path=checkpoint.path.with_suffix(".png"), show=False)
    
def load_checkpoint(filepath: Path) -> Checkpoint:
    # if model is checkpoint, load the checkpoint data
    logging.info(f"Loading checkpoint from {filepath}")
    checkpoint_data = torch.load(filepath)
    checkpoint = Checkpoint(
        path=filepath,
        model_state_dict=checkpoint_data["model_state_dict"],
        optimizer_state_dict=checkpoint_data["optimizer_state_dict"],
        params=Hyperparameters(**checkpoint_data["hyperparameters"]),
        results=TrainingResults(**checkpoint_data["results"])
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

def train(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device):
    model.train() # set to training mode
    
    n_samples = 0
    total_loss = 0
    
    for x, y in dataloader:
        
        x = x.to(device)
        y = y.to(device)
        
        pred = model(x)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # zero the gradients again so they dont sum up
        
        n_samples += x.shape[0]
        total_loss += loss.item() * x.shape[0]
        
        
    return total_loss / n_samples

def validate(model: torch.nn.Module, dataloader: DataLoader, loss_fn, device):
    model.eval() # set to evaluation mode
    
    n_samples = 0
    total_loss = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        
        n_samples += x.shape[0]
        total_loss += loss.item() * x.shape[0]
        
    return total_loss / n_samples

def optimize(checkpoint: Checkpoint):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # prepare data loaders, loss function, model, and optimizer
    train_dataloader = DataLoader(
        dataset=TODataset(Mode.TRAIN),
        batch_size=256
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(Mode.VALIDATE),
        batch_size=256
    )
    loss_fn = make_loss(checkpoint.params.loss)()
    
    
    
    # Model
    vit = ViT(
        embedding_dims=checkpoint.params.embedding_dims,
        query_dims=checkpoint.params.query_dims,
        n_attention_blocks=checkpoint.params.n_attention_blocks,
        n_attention_heads=checkpoint.params.n_attention_heads,
    ).to(device)
    logging.info(f"Initialized ViT model with embedding_dims={checkpoint.params.embedding_dims}, query_dims={checkpoint.params.query_dims}, n_attention_blocks={checkpoint.params.n_attention_blocks}, n_attention_heads={checkpoint.params.n_attention_heads}, n_parameters={vit.n_weights}")
    if checkpoint.model_state_dict is not None:
        logging.info(f"Loading model state from checkpoint: {checkpoint.path}")
        vit.load_state_dict(checkpoint.model_state_dict)
        


    optimizer = make_optimizer(checkpoint.params.optimizer)(
        params=vit.parameters(),
        lr=checkpoint.params.learning_rate
    )
    if checkpoint.optimizer_state_dict is not None:
        logging.info(f"Loading optimizer state from checkpoint: {checkpoint.path}")
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    logging.info(f"Initialized optimizer {checkpoint.params.optimizer} with lr={checkpoint.params.learning_rate}")
    
    
    
    
    # set up model directory for saving checkpoints
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "models"
    # create subdirectory for current model and checkpoints by using the model name
    model_subdir = model_dir / vit.name()
    training_subdir = model_subdir / f"{checkpoint.params.optimizer}_{checkpoint.params.learning_rate}_{checkpoint.params.loss}"
    training_subdir.mkdir(parents=True, exist_ok=True)
    
    train_losses = checkpoint.results.train_losses if checkpoint.results is not None else []
    validation_losses = checkpoint.results.validation_losses if checkpoint.results is not None else []
    
    # optimization loop
    start_epoch = len(train_losses)
    for epoch in range(start_epoch + 1, checkpoint.params.n_epochs + 1):
        
        train_loss = train(vit, train_dataloader, optimizer, loss_fn, device)
        validation_loss = validate(vit, validate_dataloader, loss_fn, device)
        
        # store training history for reporting
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        
        logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Validation Loss {validation_loss:.3f} | Delta {validation_loss-train_loss:.3f}")
        
        if epoch % 10 == 0:  # save checkpoint every 10 epochs
            
            checkpoint = Checkpoint(
                path=training_subdir / f"epoch_{epoch}.pth",
                model_state_dict=vit.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                params=checkpoint.params,
                results=TrainingResults(
                    train_losses=train_losses,
                    validation_losses=validation_losses
                ),
            )
            if epoch % 100 == 0:
                # delete previous checkpoints that are not multiples of 100
                logger.info(f"Deleting previous checkpoints that are not multiples of 100")
                for file in training_subdir.glob("epoch_*.pth"):
                    epoch_num = int(file.stem.split("_")[1])
                    if epoch_num % 100 != 0:
                        file.unlink()
            if epoch % 1000 == 0:
                logger.info(f"Deleting previous checkpoints that are not multiples of 1000")
                # delete previous checkpoints that are not multiples of 1000
                for file in training_subdir.glob("epoch_*.pth"):
                    epoch_num = int(file.stem.split("_")[1])
                    if epoch_num % 1000 != 0:
                        file.unlink()
            
            save_checkpoint(checkpoint)
        
        
    # save model as final checkpoint
    checkpoint = Checkpoint(
        path=training_subdir / f"{checkpoint.params.n_epochs}.pth",
        model_state_dict=vit.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        params=checkpoint.params,
        results=TrainingResults(
            train_losses=train_losses,
            validation_losses=validation_losses
        )
    )
    save_checkpoint(checkpoint, plot=True)
    # plot_training_history(train_losses, validation_losses, path=training_subdir / "training_history.png")
    
    return checkpoint


if __name__ == "__main__":
    
    n_epochs = 1000
    
    checkpoint_file = None # Path("models/ViT_1070424_4_2_1/adam_0.001_mse/epoch_1000.pth") # None

    if checkpoint_file is not None:
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        checkpoint = load_checkpoint(checkpoint_file)
        checkpoint.params.n_epochs = n_epochs
        start_epoch = len(checkpoint.results.train_losses) if checkpoint.results is not None else 0
        print(f"Resuming training from epoch {start_epoch}")
        assert start_epoch < checkpoint.params.n_epochs
        
    else:
        # initialize a new checkpoint if no checkpoint file is provided
        params = Hyperparameters(
            n_epochs=n_epochs,
            embedding_dims=4,
            query_dims=2,
            n_attention_blocks=1,
            n_attention_heads=2,
            optimizer="adam",
            learning_rate=0.001,
            loss="mse"
        )
        checkpoint = Checkpoint(
            path=None,
            model_state_dict=None,
            optimizer_state_dict=None,
            params=params,
            results=None
        )
    

    checkpoint = optimize(checkpoint)    
    