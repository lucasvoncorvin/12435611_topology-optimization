from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

@dataclass
class ModelTrainingModeConfig:
    """
    Configurations for training a model, that are properties of the model itself.
    E.g. dropout rate: is stored in the model object and activated via model.train()
    """
    
    attention_output_dropout_rate: float = 0.0
    mlp_output_dropout_rate: float = 0.0
    
    # Hidden layer dropout rates for attention
    # Just in case, but wont put too much effort into tuning these
    attention_hidden_dropout_rate: float = 0.0


@dataclass
class ModelSpec:
    model_name: str
    embedding_dims: int
    query_dims: int
    n_attention_blocks: int
    n_attention_heads: int
    patch_size: int
    patch_overlap: int
    training_mode_config: ModelTrainingModeConfig | None = None # optional dropout configurations

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
    lr_scheduler: str | None = None  # learning rate scheduler, e.g., "ReduceLROnPlateau"
    input_scaling: str = "log1p"
    orientation: str = "normalized"


@dataclass
class TrainingResults:
    train_losses: list
    validation_losses: list
    
@dataclass
class Checkpoint:
    model_spec: ModelSpec
    training_config: TrainingConfig
    epoch: int = 0
    model_state_dict: dict | None = None
    optimizer_state_dict: dict | None = None
    results: TrainingResults = field(
        default_factory=lambda: TrainingResults(train_losses=[], validation_losses=[])
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_spec": asdict(self.model_spec),
            "training_config": asdict(self.training_config),
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "results": asdict(self.results),
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> "Checkpoint":
        checkpoint_data = torch.load(path, map_location=device)
        model_spec_data = dict(checkpoint_data["model_spec"])
        if isinstance(model_spec_data.get("training_mode_config"), dict):
            model_spec_data["training_mode_config"] = ModelTrainingModeConfig(
                **model_spec_data["training_mode_config"]
            )
        return cls(
            model_spec=ModelSpec(**model_spec_data),
            training_config=TrainingConfig(**checkpoint_data["training_config"]),
            epoch=checkpoint_data["epoch"],
            model_state_dict=checkpoint_data["model_state_dict"],
            optimizer_state_dict=checkpoint_data["optimizer_state_dict"],
            results=TrainingResults(**checkpoint_data["results"]),
        )
    
    def get_model(self, device: torch.device = torch.device("cpu")) -> nn.Module:
        """Return the model described by this checkpoint on the requested device."""
        from .utils import build_model

        model = build_model(self.model_spec)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
        model.to(device)
        model.eval()
        return model


@dataclass
class ModelInput:
    stress: torch.Tensor # (64,64)
    strain_energy: torch.Tensor # (64,64)
    volume_fraction: torch.Tensor # (1,)
    load_position: torch.Tensor # (2,) normalized coordinates [x, y] within the grid
    load_vector: torch.Tensor # (2,) normalized vector [fx, fy] within the grid
    boundary_conditions: torch.Tensor # (256,) one-hot encoded boundary mask, 1 fixes both displacement components
    
@dataclass
class ModelOutput:
    density: torch.Tensor # (64,64)
    
@dataclass
class Sample:
    input: ModelInput
    output: ModelOutput

@dataclass
class BatchModelInput:
    stresses: torch.Tensor # (batch_size, 64, 64)
    strain_energies: torch.Tensor # (batch_size, 64, 64)
    volume_fractions: torch.Tensor # (batch_size, 1)
    load_positions: torch.Tensor # (batch_size, 2)
    load_vectors: torch.Tensor # (batch_size, 2)
    boundary_conditions: torch.Tensor # (batch_size, 256)
    
    def to(self, device: torch.device): # Move all tensors to the specified device
        self.stresses = self.stresses.to(device)
        self.strain_energies = self.strain_energies.to(device)
        self.volume_fractions = self.volume_fractions.to(device)
        self.load_positions = self.load_positions.to(device)
        self.load_vectors = self.load_vectors.to(device)
        self.boundary_conditions = self.boundary_conditions.to(device)
        return self
    
@dataclass
class BatchModelOutput:
    densities: torch.Tensor # (batch_size, 64, 64)

    def to(self, device: torch.device): # Move all tensors to the specified device
        self.densities = self.densities.to(device)
        return self

