from time import time

from .models.ViTV1 import ViT_V1
from .models.ViTV2 import ViT_V2
from .dataset import TODataset, Mode
from .loss import CustomLoss
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


MODEL_REGISTRY = {
    "vit_v1": ViT_V1,
    "vit_v2": ViT_V2
}

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
    patch_size: int 

@dataclass
class ModelSpec:
    model_name: str
    embedding_dims: int
    query_dims: int
    n_attention_blocks: int
    n_attention_heads: int
    patch_size: int

@dataclass
class TrainingConfig:
    optimizer: str
    learning_rate: float
    batch_size: int
    loss: str
    n_epochs: int # max number of epochs
@dataclass
class TrainingResults:
    train_losses: list
    validation_losses: list
@dataclass
class Checkpoint:
    path: Path | None  # path to the checkpoint file, None if starting from scratch
    model_spec: ModelSpec
    training_config: TrainingConfig
    epoch: int | None  # current epoch, None if starting from scratch
    model_state_dict: dict | None  # none if starting from scratch
    optimizer_state_dict: dict | None 
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
    logging.info(f"Saving checkpoint with to {checkpoint.path}")
    torch.save({
        "model_spec": dataclasses.asdict(checkpoint.model_spec),
        "training_config": dataclasses.asdict(checkpoint.training_config),
        "epoch": checkpoint.epoch,
        "model_state_dict": checkpoint.model_state_dict,
        "optimizer_state_dict": checkpoint.optimizer_state_dict,
        "results": dataclasses.asdict(checkpoint.results),
    }, checkpoint.path)
    if plot:
        plot_training_history(checkpoint.results.train_losses, checkpoint.results.validation_losses, path=checkpoint.path.with_suffix(".png"), show=False)
    
def load_checkpoint(filepath: Path, device = "cpu") -> Checkpoint:
    # if model is checkpoint, load the checkpoint data
    logging.info(f"Loading checkpoint from {filepath}")
    checkpoint_data = torch.load(filepath, map_location=device)
    checkpoint = Checkpoint(
        path=filepath,
        model_spec=ModelSpec(**checkpoint_data["model_spec"]),
        training_config=TrainingConfig(**checkpoint_data["training_config"]),
        epoch=checkpoint_data["epoch"],
        model_state_dict=checkpoint_data["model_state_dict"],
        optimizer_state_dict=checkpoint_data["optimizer_state_dict"],
        results=TrainingResults(**checkpoint_data["results"])
    )
    return checkpoint

def make_optimizer(optimizer_name: str):
    """Optimizer factory
    
    returns the corresponding optimizer class based on the optimizer_name provided.
    """
    if optimizer_name == "adam":
        return optim.Adam
    if optimizer_name == "adadelta":
        return optim.Adadelta
    if optimizer_name == "adamw":
        return optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def make_loss(loss_name: str, patch_size=None):
    """Loss function factory
    
    
    
    returns the corresponding loss function class based on the loss_name provided.
    """
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "kl_divergence":
        return nn.KLDivLoss()
    if loss_name == "bce":
        return nn.BCELoss()
    if loss_name == "custom_mse":
        if not patch_size:
            raise ValueError("patch_size must be provided for custom loss functions")
        return CustomLoss(base_loss_fn=nn.MSELoss(), patch_size=patch_size)
    if loss_name == "custom_bce":
        if not patch_size:
            raise ValueError("patch_size must be provided for custom loss functions")
        return CustomLoss(base_loss_fn=nn.BCELoss(), patch_size=patch_size)
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
        if isinstance(loss_fn, CustomLoss):
            load = x[:, 1:5]
            target_volume_fraction = x[:, 0]
            loss = loss_fn(pred, y, load, target_volume_fraction)
        else:
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
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            if isinstance(loss_fn, CustomLoss):
                load = x[:, 1:5]
                target_volume_fraction = x[:, 0]
                loss = loss_fn(pred, y, load, target_volume_fraction)
            else:
                loss = loss_fn(pred, y)
            
            n_samples += x.shape[0]
            total_loss += loss.item() * x.shape[0]
        
    return total_loss / n_samples

def optimize(checkpoint: Checkpoint, training_data_dir: Path, validation_data_dir: Path):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # prepare data loaders, loss function, model, and optimizer
    train_dataloader = DataLoader(
        dataset=TODataset(training_data_dir, checkpoint.model_spec.patch_size),
        batch_size=checkpoint.training_config.batch_size
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(validation_data_dir, checkpoint.model_spec.patch_size),
        batch_size=checkpoint.training_config.batch_size
    )
    loss_fn = make_loss(checkpoint.training_config.loss, patch_size=checkpoint.model_spec.patch_size)
    
    
    
    # Model
    vit = MODEL_REGISTRY[checkpoint.model_spec.model_name](
        embedding_dims=checkpoint.model_spec.embedding_dims,
        query_dims=checkpoint.model_spec.query_dims,
        n_attention_blocks=checkpoint.model_spec.n_attention_blocks,
        n_attention_heads=checkpoint.model_spec.n_attention_heads,
        patch_size=checkpoint.model_spec.patch_size
    ).to(device)
    logging.info(f"Initialized ViT model with embedding_dims={checkpoint.model_spec.embedding_dims}, query_dims={checkpoint.model_spec.query_dims}, n_attention_blocks={checkpoint.model_spec.n_attention_blocks}, n_attention_heads={checkpoint.model_spec.n_attention_heads}, n_parameters={vit.n_weights}")
    if checkpoint.model_state_dict is not None:
        logging.info(f"Loading model state from checkpoint: {checkpoint.path}")
        vit.load_state_dict(checkpoint.model_state_dict)
        


    optimizer = make_optimizer(checkpoint.training_config.optimizer)(
        params=vit.parameters(),
        lr=checkpoint.training_config.learning_rate
    )
    if checkpoint.optimizer_state_dict is not None:
        logging.info(f"Loading optimizer state from checkpoint: {checkpoint.path}")
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    logging.info(f"Initialized optimizer {checkpoint.training_config.optimizer} with lr={checkpoint.training_config.learning_rate}")
    
    
    
    
    # set up model directory for saving checkpoints
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "trainings"
    # create subdirectory for current model and checkpoints by using the model name
    model_subdir = model_dir / vit.name()
    training_subdir = model_subdir / f"{checkpoint.training_config.optimizer}_{checkpoint.training_config.learning_rate}_{checkpoint.training_config.loss}"
    training_subdir.mkdir(parents=True, exist_ok=True)
    
    train_losses = checkpoint.results.train_losses if checkpoint.results is not None else []
    validation_losses = checkpoint.results.validation_losses if checkpoint.results is not None else []
    
    # optimization loop
    start_epoch = len(train_losses)
    for epoch in range(start_epoch + 1, checkpoint.training_config.n_epochs + 1):
        start_time = time()
        
        train_loss = train(vit, train_dataloader, optimizer, loss_fn, device)
        validation_loss = validate(vit, validate_dataloader, loss_fn, device)
        
        # store training history for reporting
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        
        logger.info(f"Epoch: {epoch} | Time: {time() - start_time:.2f}s | Train Loss: {train_loss:.5f} | Validation Loss {validation_loss:.5f} | Delta {validation_loss-train_loss:.5f}")
        
        if epoch % 10 == 0:  # save checkpoint every 10 epochs
            
            checkpoint = Checkpoint(
                path=training_subdir / f"epoch_{epoch}.pth",
                model_spec=checkpoint.model_spec,
                epoch=epoch,
                model_state_dict=vit.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                training_config=checkpoint.training_config,
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
            # if epoch % 1000 == 0:
            #     logger.info(f"Deleting previous checkpoints that are not multiples of 1000")
            #     # delete previous checkpoints that are not multiples of 1000
            #     for file in training_subdir.glob("epoch_*.pth"):
            #         epoch_num = int(file.stem.split("_")[1])
            #         if epoch_num % 1000 != 0:
            #             file.unlink()
            
            save_checkpoint(checkpoint)
        
        
    # save model as final checkpoint
    checkpoint = Checkpoint(
        path=training_subdir / f"{checkpoint.training_config.n_epochs}.pth",
        model_spec=checkpoint.model_spec,
        epoch=checkpoint.training_config.n_epochs,
        model_state_dict=vit.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        training_config=checkpoint.training_config,
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
    
    checkpoint_file = None # Path("trainings/ViT_V1_e4_q2_b1_h2_p8/adam_0.001_mse/epoch_10.pth") # None

    project_root = Path(__file__).parent.parent
    training_data_dir = project_root / "data/labeled_data/train/augmented_mirrored"
    validation_data_dir = project_root / "data/labeled_data/val"

    if checkpoint_file is not None:
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        checkpoint = load_checkpoint(checkpoint_file)
        checkpoint.training_config.n_epochs = n_epochs
        start_epoch = checkpoint.epoch if checkpoint.results is not None else 0
        logger.info(f"Resuming training from epoch {start_epoch}")
        assert start_epoch is not None and start_epoch < checkpoint.training_config.n_epochs
        
    else:
        # initialize a new checkpoint if no checkpoint file is provided
        model_spec = ModelSpec(
            model_name="vit_v2",
            embedding_dims=128,
            query_dims=64,
            n_attention_blocks=6, # layers, I dont know why I called them blocks
            n_attention_heads=4,
            patch_size=8
        )
        
        assert model_spec.model_name in MODEL_REGISTRY.keys()
        
        training_config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=256,
            optimizer="adam",
            learning_rate=0.001,
            loss="bce",
        )
        checkpoint = Checkpoint(
            path=None,
            epoch=0,
            model_spec=model_spec,
            model_state_dict=None,
            optimizer_state_dict=None,
            training_config=training_config,
            results=None
        )
    

    checkpoint = optimize(checkpoint, training_data_dir, validation_data_dir)    
    