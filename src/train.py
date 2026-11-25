from time import time
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import TODataset, Mode, orientation_for_evaluation
from .loss import BinaryMissmatchLoss, CustomLoss, ModularLoss, SinkhornDistance, VolumeFractionLoss
from .utils import MODEL_REGISTRY, build_model, collate_samples
from .model_types import ModelSpec, ModelTrainingModeConfig, Checkpoint, TrainingResults, TrainingConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    
def make_lr_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer, **kwargs):
    """Learning rate scheduler factory
    
    returns the corresponding learning rate scheduler based on the scheduler_name provided.
    """
    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

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
    if loss_name.startswith("mse_binary_mismatch_"): # MSE loss combined with binary mismatch
        weight_str = loss_name.split("_")[-1] 
        if weight_str not in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            raise ValueError(f"Invalid weight for mse_binary_mismatch: {weight_str}")
        try:
            weight = float(weight_str) / 10
        except ValueError:
            raise ValueError(f"Invalid weight for mse_binary_mismatch: {weight_str}")
        return ModularLoss(
            loss_fns=torch.nn.ModuleList([nn.MSELoss(), BinaryMissmatchLoss()]),
            weights=[1-weight, weight]
        )
    if loss_name == "mse_bmvf_equal": # MSE loss combined with binary mismatch and volume fraction
        return ModularLoss(
            loss_fns=torch.nn.ModuleList([nn.MSELoss(), BinaryMissmatchLoss(), VolumeFractionLoss()]),
            weights=[1/3, 1/3, 1/3]
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
        elif isinstance(loss_fn, ModularLoss):
            loss = loss_fn(pred.densities, y.densities, target_volume_fraction=x.volume_fractions)
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
            elif isinstance(loss_fn, ModularLoss):
                loss = loss_fn(pred.densities, y.densities, target_volume_fraction=x.volume_fractions)
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
        dataset=TODataset(
            npz_path,
            Mode.TRAIN,
            input_scaling=checkpoint.training_config.input_scaling,
            orientation=checkpoint.training_config.orientation,
        ),
        batch_size=checkpoint.training_config.batch_size,
        collate_fn=collate_samples,
        shuffle=True,
    )
    validate_dataloader = DataLoader(
        dataset=TODataset(
            npz_path,
            Mode.VALIDATE,
            input_scaling=checkpoint.training_config.input_scaling,
            orientation=orientation_for_evaluation(checkpoint.training_config.orientation),
        ),
        batch_size=checkpoint.training_config.batch_size,
        collate_fn=collate_samples
    )
    
    # Loss
    loss_fn = make_loss(checkpoint.training_config.loss, patch_size=checkpoint.model_spec.patch_size).to(device)
    
    # Model
    vit = build_model(checkpoint.model_spec).to(device)
    
    # extract dropout rates for logging purposes (they are not used for the older models)
    attention_output_dropout_rate = getattr(vit, "attention_output_dropout_rate", 0.0)
    mlp_output_dropout_rate = getattr(vit, "mlp_output_dropout_rate", 0.0)
    attention_hidden_dropout_rate = getattr(vit, "attention_hidden_dropout_rate", 0.0)
    
    logging.info(f"""Initialized ViT model with 
                 embedding_dims={checkpoint.model_spec.embedding_dims}, 
                 query_dims={checkpoint.model_spec.query_dims}, 
                 n_attention_blocks={checkpoint.model_spec.n_attention_blocks}, 
                 n_attention_heads={checkpoint.model_spec.n_attention_heads}, 
                 n_parameters={vit.n_weights}, 
                 attention_output_dropout_rate={attention_output_dropout_rate},
                 mlp_output_dropout_rate={mlp_output_dropout_rate},
                 attention_hidden_dropout_rate={attention_hidden_dropout_rate}""")
    if checkpoint.model_state_dict is not None:
        logging.info("Loading model state from checkpoint.")
        vit.load_state_dict(checkpoint.model_state_dict)
        

    # Optimizer
    optimizer = make_optimizer(checkpoint.training_config.optimizer)(
        params=vit.parameters(),
        lr=checkpoint.training_config.learning_rate
    )
    if checkpoint.optimizer_state_dict is not None:
        logging.info("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    lr_scheduler = None
    if checkpoint.training_config.lr_scheduler is not None:
        lr_scheduler = make_lr_scheduler(checkpoint.training_config.lr_scheduler, optimizer)
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
    scheduler_string = (
        f"_{checkpoint.training_config.lr_scheduler}"
        if checkpoint.training_config.lr_scheduler is not None else ""
    )
    training_subdir = model_subdir / (
        f"{checkpoint.training_config.optimizer}"
        f"_{checkpoint.training_config.learning_rate}"
        f"{scheduler_string}"
        f"_{checkpoint.training_config.loss}"
        f"_datt{attention_output_dropout_rate}"
        f"_dmlp{mlp_output_dropout_rate}"
    )
    training_subdir.mkdir(parents=True, exist_ok=True)
    
    train_losses = checkpoint.results.train_losses
    validation_losses = checkpoint.results.validation_losses
    
    # select random examples from the validation dataset for TensorBoard visualization
    example_indices = torch.randperm(
        len(validate_dataloader.dataset),
        generator=torch.Generator().manual_seed(42), # fix a seed for reproducibility
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
            start_time = time() # for logging epoch duration
            
            train_loss = train(vit, train_dataloader, optimizer, loss_fn, device)
            validation_loss = validate(vit, validate_dataloader, loss_fn, device)
            if lr_scheduler is not None:
                lr_scheduler.step(validation_loss)
            
            # tensorboard logging for validation examples
            if epoch % 10 == 0:
                log_validation_examples(writer, vit, examples, epoch, device)
            
            # store training history for reporting
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", validation_loss, epoch)
            
            logger.info(f"Epoch: {epoch} | Time: {time() - start_time:.2f}s | Train Loss: {train_loss:.5f} | Validation Loss {validation_loss:.5f} | Delta {validation_loss-train_loss:.5f} | LR Scheduler: {checkpoint.training_config.lr_scheduler if checkpoint.training_config.lr_scheduler is not None else 'None'}")
            
            # some checkpoints at specific epochs
            checkpoint = Checkpoint(
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
            checkpoint.save(training_subdir / "latest.pth")
            # save every 50th epoch by default (to compare progress)
            if epoch % 50 == 0:
                epoch_checkpoint_path = training_subdir / f"epoch_{epoch}.pth"
                logging.info("Saving checkpoint to %s", epoch_checkpoint_path)
                checkpoint.save(epoch_checkpoint_path)
            
            # save best checkpoint based on validation loss
            if validation_loss < best_validation_loss and epoch > 10: # first 10 epochs are useless (i think)
                best_validation_loss = validation_loss
                best_checkpoint_path = training_subdir / "best.pth"
                logging.info("Saving checkpoint to %s", best_checkpoint_path)
                checkpoint.save(best_checkpoint_path)
            
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
        logging.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = Checkpoint.load(checkpoint_file)
        start_epoch = checkpoint.epoch
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
            model_name="vit_v1",
            embedding_dims=4,
            query_dims=2,
            n_attention_blocks=1,
            n_attention_heads=2,
            patch_size=8,
            patch_overlap=0,
        )
        
        assert model_spec.model_name in MODEL_REGISTRY.keys()
        
        training_config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=256,
            optimizer="adam",
            learning_rate=0.001,
            loss="mse",
            input_scaling="raw",
            orientation="augment",
            # lr_scheduler="ReduceLROnPlateau"
        )
        checkpoint = Checkpoint(
            epoch=0,
            model_spec=model_spec,
            model_state_dict=None,
            optimizer_state_dict=None,
            training_config=training_config,
        )
    

    checkpoint = optimize(checkpoint, npz_path, custom_training_subdir=custom_training_subdir)
    