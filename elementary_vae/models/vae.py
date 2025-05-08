import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Implement the VAE encoder layers here
        # Unlike a regular autoencoder, a VAE encoder outputs
        # both a mean (mu) and log variance (logvar) for each
        # latent dimension
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward pass
        # Remember to flatten the input image first
        # Return both mu and logvar for the latent space
        
        return None, None  # Replace with mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = [256, 512], output_dim: int = 784) -> None:
        super().__init__()
        
        # TODO: Implement the VAE decoder layers here
        # The decoder takes a sampled latent vector and
        # reconstructs the original input
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to reshape the output to match the original image dimensions
        
        return None  # Replace with your reconstruction

class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Create the encoder and decoder components
        self.encoder: Optional[VAEEncoder] = None  # Replace with your VAE encoder
        self.decoder: Optional[VAEDecoder] = None  # Replace with your decoder
        self.latent_dim: int = latent_dim
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the reparameterization trick
        # This samples from a distribution defined by mu and logvar
        # while allowing gradients to flow back
        
        return None  # Replace with the sampled latent vector
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement the full VAE forward pass
        # 1. Encode the input to get mu, logvar
        # 2. Sample from the distribution using reparameterize
        # 3. Decode the sampled latent vector
        
        return None, None, None  # Replace with reconstruction, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement encoding and sampling
        
        return None
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement decoding only
        
        return None
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # TODO: Generate samples from random points in the latent space
        
        return None  # Replace with your samples 