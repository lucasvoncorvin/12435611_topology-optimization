from dataclasses import dataclass

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
