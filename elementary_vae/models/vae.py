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
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.fc4 = nn.Linear(hidden_dims[1], latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward pass
        # Remember to flatten the input image first
        # Return both mu and logvar for the latent space
        x = x.view(x.size(0), -1)
        o = self.relu(self.fc1(x))
        o = self.relu(self.fc2(o))
        mu = self.fc3(o)
        logvar = self.fc4(o)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = [256, 512], output_dim: int = 784) -> None:
        super().__init__()
        
        # TODO: Implement the VAE decoder layers here
        # The decoder takes a sampled latent vector and
        # reconstructs the original input
        self.fc1 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to reshape the output to match the original image dimensions
        o = self.relu(self.fc1(z))
        o = self.relu(self.fc2(o))
        o = torch.sigmoid(self.fc3(o))
        return o.view(-1, 1, 28, 28) # hardcoded for MNIST

class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Create the encoder and decoder components
        rev_hidden_dims = hidden_dims[::-1]
        self.encoder: Optional[VAEEncoder] = VAEEncoder(input_dim, hidden_dims, latent_dim)  # Replace with your VAE encoder
        self.decoder: Optional[VAEDecoder] = VAEDecoder(latent_dim, rev_hidden_dims, input_dim)  # Replace with your decoder
        self.latent_dim: int = latent_dim
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the reparameterization trick
        # This samples from a distribution defined by mu and logvar
        # while allowing gradients to flow back
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # sample from standard normal
        return mu + eps * std  # Replace with the sampled latent vector
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement the full VAE forward pass
        # 1. Encode the input to get mu, logvar
        # 2. Sample from the distribution using reparameterize
        # 3. Decode the sampled latent vector
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement encoding and sampling
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement decoding only
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # TODO: Generate samples from random points in the latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)  # Replace with your samples 