from time import time
from pathlib import Path
import logging
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .models.ViTV1 import ViT_V1
from .models.ViTV2 import ViT_V2
from .models.ViTV3 import ViT_V3
from .models.ViTV4 import ViT_V4
from .dataset import TODataset, Mode
from .loss import BinaryMissmatchLoss, CustomLoss, ModularLoss, SinkhornDistance
from .utils import collate_samples
from .types import ModelSpec, ModelTrainingModeConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_REGISTRY = {
    "vit_v1": ViT_V1,
    "vit_v2": ViT_V2,
    "vit_v3": ViT_V3,
    "vit_v4": ViT_V4
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
    patch_overlap: int

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

def save_checkpoint(checkpoint: Checkpoint, plot: bool = False, log: bool = True):
    if checkpoint.path is None:
        raise ValueError("Checkpoint path is None. Cannot save checkpoint.")
    if checkpoint.results is None:
        checkpoint.results = TrainingResults(train_losses=[], validation_losses=[])
        logger.warning("Checkpoint results are None.")
    checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
    if log:
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
    model_spec_data = dict(checkpoint_data["model_spec"])
    if isinstance(model_spec_data.get("training_mode_config"), dict):
        model_spec_data["training_mode_config"] = ModelTrainingModeConfig(**model_spec_data["training_mode_config"])
    checkpoint = Checkpoint(
        path=filepath,
        model_spec=ModelSpec(**model_spec_data),
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
    if loss_name == "wasserstein":
        if not patch_size:
            raise ValueError("patch_size must be provided for custom loss functions")
        return SinkhornDistance(patch_size=patch_size)
    if loss_name == "mse_binary_mismatch_01": # MSE loss combined with 10% binary mismatch
        return ModularLoss(
            loss_fns=torch.nn.ModuleList([nn.MSELoss(), BinaryMissmatchLoss()]),
            weights=[0.9, 0.1]
        )
    
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
            loss = loss_fn(pred.densities, y.densities)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # zero the gradients again so they dont sum up
        
        n_samples += x.stresses.shape[0]
        total_loss += loss.item() * x.stresses.shape[0]
        
        
    return total_loss / n_samples

def validate(model: torch.nn.Module, dataloader: DataLoader, loss_fn, device):
    model.eval() # set to evaluation mode
    
    n_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_input, batch_output in dataloader:
            x = batch_input.to(device)
            y = batch_output.to(device)

            pred = model(x)
            if isinstance(loss_fn, CustomLoss):
                load = torch.stack([batch_input.load_positions, batch_input.load_vectors], dim=-1)
                target_volume_fraction = batch_input.volume_fractions
                loss = loss_fn(pred, y, load, target_volume_fraction)
            else:
                loss = loss_fn(pred.densities, y.densities)
            
            n_samples += x.stresses.shape[0]
            total_loss += loss.item() * x.stresses.shape[0]
        
    return total_loss / n_samples

def log_validation_examples(writer, model, examples, epoch, device):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            predictions = model(examples[0].to(device)).densities.cpu()
    finally:
        model.train(was_training)

    targets = examples[1].densities
    writer.add_images("Validation/random_samples/prediction", predictions.unsqueeze(1), epoch)
    writer.add_images("Validation/random_samples/absolute_error", (predictions - targets).abs().unsqueeze(1), epoch)
    writer.add_histogram("Validation/random_samples/predicted_density", predictions, epoch)

def optimize(checkpoint: Checkpoint, npz_path: Path, custom_training_subdir: Path | None = None):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Data
    # prepare data loaders, loss function, model, and optimizer
    train_dataloader = DataLoader(
        dataset=TODataset(npz_path, Mode.TRAIN),
        batch_size=checkpoint.training_config.batch_size,
        collate_fn=collate_samples,
        shuffle=True,
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(npz_path, Mode.VALIDATE),
        batch_size=checkpoint.training_config.batch_size,
        collate_fn=collate_samples
    )
    
    # Loss
    loss_fn = make_loss(checkpoint.training_config.loss, patch_size=checkpoint.model_spec.patch_size).to(device)
    
    # Model
    vit = MODEL_REGISTRY[checkpoint.model_spec.model_name](
       checkpoint.model_spec
    ).to(device)
    
    logging.info(f"""Initialized ViT model with 
                 embedding_dims={checkpoint.model_spec.embedding_dims}, 
                 query_dims={checkpoint.model_spec.query_dims}, 
                 n_attention_blocks={checkpoint.model_spec.n_attention_blocks}, 
                 n_attention_heads={checkpoint.model_spec.n_attention_heads}, 
                 n_parameters={vit.n_weights}, 
                 attention_output_dropout_rate={vit.attention_output_dropout_rate}, 
                 mlp_output_dropout_rate={vit.mlp_output_dropout_rate}, 
                 attention_hidden_dropout_rate={vit.attention_hidden_dropout_rate}""")
    if checkpoint.model_state_dict is not None:
        logging.info(f"Loading model state from checkpoint: {checkpoint.path}")
        vit.load_state_dict(checkpoint.model_state_dict)
        

    # Optimizer
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
    if custom_training_subdir is not None:
        model_subdir = custom_training_subdir
    if not model_subdir.exists():
        model_subdir.mkdir(parents=True, exist_ok=True)
    training_subdir = model_subdir / (
        f"{checkpoint.training_config.optimizer}"
        f"_{checkpoint.training_config.learning_rate}"
        f"_{checkpoint.training_config.loss}"
        f"_datt{vit.attention_output_dropout_rate}"
        f"_dmlp{vit.mlp_output_dropout_rate}"
    )
    training_subdir.mkdir(parents=True, exist_ok=True)
    
    train_losses = checkpoint.results.train_losses if checkpoint.results is not None else []
    validation_losses = checkpoint.results.validation_losses if checkpoint.results is not None else []
    
    # select random examples from the validation dataset for TensorBoard visualization
    example_indices = torch.randperm(
        len(validate_dataloader.dataset),
        generator=torch.Generator().manual_seed(42),
    )[:16].tolist()
    if not example_indices:
        raise ValueError("Cannot select TensorBoard examples from an empty validation dataset.")
    examples = collate_samples([
        validate_dataloader.dataset[i]
        for i in example_indices
    ])
    
    # optimization loop
    start_epoch = len(train_losses)
    writer = SummaryWriter(
        log_dir=training_subdir / "tensorboard",
        purge_step=start_epoch +1
    )
    try:
        writer.add_text("Validation/random_samples/indices", f"Validation indices: {example_indices}", 0)
        writer.add_images("Validation/random_samples/target", examples[1].densities.unsqueeze(1), 0)
        best_validation_loss = min(validation_losses) if validation_losses else float("inf")
        for epoch in range(start_epoch + 1, checkpoint.training_config.n_epochs + 1):
            start_time = time()
            
            train_loss = train(vit, train_dataloader, optimizer, loss_fn, device)
            validation_loss = validate(vit, validate_dataloader, loss_fn, device)
            
            # tensorboard logging for validation examples
            if epoch % 10 == 0:
                log_validation_examples(writer, vit, examples, epoch, device)
            
            # store training history for reporting
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", validation_loss, epoch)
            
            logger.info(f"Epoch: {epoch} | Time: {time() - start_time:.2f}s | Train Loss: {train_loss:.5f} | Validation Loss {validation_loss:.5f} | Delta {validation_loss-train_loss:.5f}")
            
            # some checkpoints at specific epochs
            checkpoint = Checkpoint(
                path=training_subdir / f"latest.pth",
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
            save_checkpoint(checkpoint, log=False)
            # save every 50th epoch by default (to compare progress)
            if epoch % 50 == 0:
                checkpoint.path = training_subdir / f"epoch_{epoch}.pth"
                save_checkpoint(checkpoint)
            
            # save best checkpoint based on validation loss
            if validation_loss < best_validation_loss and epoch > 10: # first 10 epochs are useless (i think)
                best_validation_loss = validation_loss
                checkpoint.path = training_subdir / f"best.pth"
                save_checkpoint(checkpoint) 
            
    finally:
        writer.close()
        
            
    return checkpoint


if __name__ == "__main__":
    
    n_epochs = 500
    
    checkpoint_file = None  # Path("trainings/ViT_V4_e128_b9_h4_p16_c2_o2/adam_0.0003_mse_datt0.1_dmlp0.1/epoch_60.pth") # None
    
    # ignore
    custom_training_subdir: Path | None = None
    next_stage: TrainingConfig | None = None
    # ignore

    project_root = Path(__file__).parent.parent
    npz_path = project_root / "data" / "processed" / "topopt_data.npz"
    
    if checkpoint_file is not None:
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        checkpoint = load_checkpoint(checkpoint_file)
        start_epoch = checkpoint.epoch if checkpoint.results is not None else 0
        logger.info(f"Resuming training from epoch {start_epoch}")

        # next_stage = TrainingConfig(
        #     n_epochs=n_epochs,
        #     batch_size=256,
        #     optimizer="adam",
        #     learning_rate=0.0003,
        #     loss="mse"
        # )
        if next_stage is None:
            checkpoint.training_config.n_epochs = n_epochs
        else:
            checkpoint.training_config = next_stage
            checkpoint.optimizer_state_dict = None

            # Store a new stage in a nested directory below the parent checkpoint.
            custom_training_subdir = checkpoint_file.parent / f"from_{checkpoint.epoch}"

        assert start_epoch is not None and start_epoch < checkpoint.training_config.n_epochs
    else:
        # initialize a new checkpoint if no checkpoint file is provided
        model_spec = ModelSpec(
            model_name="vit_v4",
            embedding_dims=128,
            query_dims=128,
            n_attention_blocks=6, # layers, I dont know why I called them blocks
            n_attention_heads=4,
            patch_size=16,
            patch_overlap=2,
            training_mode_config=ModelTrainingModeConfig(
                attention_output_dropout_rate=0.1,
                mlp_output_dropout_rate=0.1,
                attention_hidden_dropout_rate=0.0,
            )
        )
        
        assert model_spec.model_name in MODEL_REGISTRY.keys()
        
        training_config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=256,
            optimizer="adam",
            learning_rate=0.0003,
            loss="mse_binary_mismatch_01",
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
    

    checkpoint = optimize(checkpoint, npz_path, custom_training_subdir=custom_training_subdir)
    