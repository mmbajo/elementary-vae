import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

class Encoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Implement the encoder layers here
        # You should create layers to process input data
        # and project it to the latent space
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to flatten the input image first
        
        return None  # Replace with your latent representation

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = [256, 512], output_dim: int = 784) -> None:
        super().__init__()
        
        # TODO: Implement the decoder layers here
        # You should create layers to process the latent vector
        # and reconstruct the original input
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to reshape the output to match the original image dimensions
        
        return None  # Replace with your reconstruction

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Create the encoder and decoder components
        self.encoder: Optional[Encoder] = None  # Replace with your encoder
        self.decoder: Optional[Decoder] = None  # Replace with your decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the full autoencoder forward pass
        # 1. Encode the input
        # 2. Decode the latent representation
        
        return None  # Replace with the reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement encoding only
        return None
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement decoding only
        return None 