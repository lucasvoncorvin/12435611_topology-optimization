import torch
import torch.nn as nn
import math
from .utils import add_positional_encoding, attention, build_positional_encoding, build_parameter_vector, patchify
from ..utils import BatchModelInput, BatchModelOutput


class ViT_V3(torch.nn.Module):
    """
    Changelog:
    
    Use pytorch implementation of multihead attention
    """
    def __init__(
            self,
            embedding_dims,
            query_dims,
            n_attention_blocks,
            n_attention_heads,
            patch_side_lenth,
            n_channels,
        ):
        
        assert __file__.endswith(self.__class__.__name__[-2::] + ".py") # just check I actually made sure to align the versions (up to V9)
        assert embedding_dims % n_attention_heads == 0, "Embedding dimensions must be divisible by the number of attention heads"
        assert query_dims % n_attention_heads == 0, "Query dimensions must be divisible by the number of attention heads"
        
        super(ViT_V3, self).__init__()
        
        # create seed for reproducibility
        torch.manual_seed(42)
        
        # hyperparameters
        self.embedding_dims = embedding_dims
        self.query_dims = query_dims
        self.n_attention_blocks = n_attention_blocks
        self.n_attention_heads = n_attention_heads
        
        self.n_patches = (64 // patch_side_lenth) * (64 // patch_side_lenth) 
        self.patch_side_lenth = patch_side_lenth
        self.pixels_per_patch = self.patch_side_lenth**2
        self.n_channels = n_channels
        
        self.embedding_dims_head = self.embedding_dims // self.n_attention_heads
        self.query_dims_head = self.query_dims // self.n_attention_heads
        
        # input and output
        self.linear_patch = nn.Linear(self.pixels_per_patch*self.n_channels, self.embedding_dims)
        self.linear_param = nn.Linear(1+4+4*64, self.embedding_dims) # dimension of the parameter vector
        self.readout = nn.Linear(self.embedding_dims, self.pixels_per_patch) # get the whole optimized topology at once
        
        # attention stuff
        
        
        attention_norms = []
        mlps = []
        mlp_norms = []
        multihead_attentions = []
        for block in range(self.n_attention_blocks):            
            
            multihead_attentions.append(nn.MultiheadAttention(
                embed_dim=self.embedding_dims,
                num_heads=self.n_attention_heads,
                batch_first=True,
            ))
            
            mlps.append(nn.Sequential(
                nn.Linear(self.embedding_dims, 4*self.embedding_dims), # researched a bit and that this is often 4 times the input of the mlp
                nn.ReLU(),
                nn.Linear(4*self.embedding_dims, self.embedding_dims)
            )) # non-linear projection for each attention block
            
            attention_norms.append(nn.LayerNorm(self.embedding_dims)) # create separate layer norms for each attention block
            mlp_norms.append(nn.LayerNorm(self.embedding_dims)) # create separate layer norms for each MLP block
            
        self.mlps = nn.ModuleList(mlps)
        self.multihead_attentions = nn.ModuleList(multihead_attentions)
        self.attention_norms = nn.ModuleList(attention_norms)
        self.mlp_norms = nn.ModuleList(mlp_norms)
        self.final_norm = nn.LayerNorm(self.embedding_dims)
        
        
        # positional encodings 
        self.register_buffer(
            "positional_encoding",
            build_positional_encoding(patch_side_lenth, embedding_dims),
            persistent=False,
        )
        
        
        self.n_weights = sum(p.numel() for p in self.parameters() if p.requires_grad) # total number of trainable parameters
                
        
    def forward(self, batch: BatchModelInput) -> BatchModelOutput:
        """Forward pass of the Vision Transformer.

        Args:
            batch (BatchModelInput): Input batch containing stresses, strain_energies, volume_fractions, load_positions, load_directions, and boundary_conditions.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches*elements_per_patch)
            Each output corresponds to the concatenated flattened patches of densities in the optimized topology.
        """
        n_samples = batch.stresses.shape[0]
        
        params = build_parameter_vector(
            load_positions=batch.load_positions,
            load_vectors=batch.load_vectors,
            boundary_conditions=batch.boundary_conditions,
            volume_fractions=batch.volume_fractions
        ) # -> (batch_size, 1 + 4 + 4*64) = (batch_size, 261)

        # skip creating multi-channel image for now, using stress field as input directly
        images = torch.stack([batch.stresses, batch.strain_energies], dim=-1)

        patches = patchify(
            images=images, # for we can use the stress field as the input image directly, but if we want more fields we must create the multi-channel image first
            meta_shape=(64 // self.patch_side_lenth, 64 // self.patch_side_lenth), 
            overlap=0
        ) # -> (batch_size, n_patches_x, n_patches_y, patch_height, patch_width, channels)
        
        tokens = patches.view(
            n_samples,
            patches.shape[1] * patches.shape[2],  # n_patches_x * n_patches_y
            patches.shape[3] * patches.shape[4] * patches.shape[5]  # flatten patch_height, patch_width, channels
        )
    
        # embedding of parameters (load, boundary conditions, volume fraction)
        param_embeddings = self.linear_param(params) 
        
        # embedding of patches
        token_embeddings = self.linear_patch(tokens) + self.positional_encoding
        
        embeddings = torch.cat([param_embeddings.unsqueeze(1), token_embeddings], dim=1)
        
        for i in range(self.n_attention_blocks):
            normalized = self.attention_norms[i](embeddings)

            attn_out, _ = self.multihead_attentions[i](normalized, normalized, normalized, need_weights=False)
            embeddings = embeddings + attn_out
            
            embeddings = embeddings + self.mlps[i](self.mlp_norms[i](embeddings)) # apply MLP with its own layer norm
        
        # readout
        embeddings = self.final_norm(embeddings)
        res = self.readout(embeddings[:, 1:, :]) # readout for each token embedding
        # res: (batch_size, n_tokens, patch_height * patch_width)
        
        # reshape the readout to create density tensor of shape (batch_size, height, width)
        # i.e. unflatten each token and concatenate to form the full image
        self.n_patches_x = 64 // self.patch_side_lenth
        self.n_patches_y = 64 // self.patch_side_lenth
        patch_height = self.patch_side_lenth
        patch_width = self.patch_side_lenth

        res = res.view(
            n_samples,
            self.n_patches_x,
            self.n_patches_y,
            patch_height,
            patch_width
        )

        res = res.permute(0, 1, 3, 2, 4).contiguous() # (batch_size, n_patches_x, patch_height, n_patches_y, patch_width)
        res = res.view(n_samples, self.n_patches_x * patch_height, self.n_patches_y * patch_width) # (batch_size, height, width)
        
        output = BatchModelOutput(
            densities=torch.sigmoid(res) # map to the range [0, 1]
        )
        return output
            
    
    def name(self):
        """Defines a name based on the model architecture."""
        # self.query_dims do not affect the model name anymore
        return f"{self.__class__.__name__}_e{self.embedding_dims}_b{self.n_attention_blocks}_h{self.n_attention_heads}_p{self.patch_side_lenth}_c{self.n_channels}"
    