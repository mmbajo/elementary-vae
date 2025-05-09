import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Union


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        # Create a dynamic list of layers based on hidden_dims
        self.hidden_layers = nn.ModuleList()
        input_size = input_dim

        # Add all hidden layers
        for hidden_size in hidden_dims:
            linear = nn.Linear(input_size, hidden_size)
            # Initialize weights with Xavier/Glorot initialization
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.hidden_layers.append(linear)
            input_size = hidden_size

        # Output layers for mu and logvar
        self.mu_layer = nn.Linear(input_size, latent_dim)
        self.logvar_layer = nn.Linear(input_size, latent_dim)

        # Initialize mu layer
        nn.init.xavier_normal_(self.mu_layer.weight, gain=0.01)
        nn.init.zeros_(self.mu_layer.bias)

        # Initialize logvar layer with small values to prevent initial KL from being too large
        nn.init.xavier_normal_(self.logvar_layer.weight, gain=0.01)
        nn.init.constant_(self.logvar_layer.bias, -5)  # Start with small variance

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        # Get mu and logvar
        mu = self.mu_layer(x)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(self.logvar_layer(x), min=-10.0, max=10.0)

        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: List[int] = [256, 512],
        output_dim: int = 784,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        # Create a dynamic list of layers based on hidden_dims
        self.hidden_layers = nn.ModuleList()
        input_size = latent_dim

        # Add all hidden layers
        for hidden_size in hidden_dims:
            linear = nn.Linear(input_size, hidden_size)
            # Initialize weights with Xavier/Glorot initialization
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.hidden_layers.append(linear)
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(input_size, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.relu = nn.ReLU()

        # Store output dimensions for reshaping
        self.output_dim = output_dim
        self.image_shape = image_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Pass through hidden layers
        x = z
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        # Output layer with sigmoid activation
        output = torch.sigmoid(self.output_layer(x))

        # Reshape to match original image dimensions
        return output.view(output.size(0), *self.image_shape)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 32,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        # Create encoder and decoder with reversed hidden dims for decoder
        rev_hidden_dims = hidden_dims[::-1]
        self.encoder: Optional[VAEEncoder] = VAEEncoder(
            input_dim, hidden_dims, latent_dim
        )
        self.decoder: Optional[VAEDecoder] = VAEDecoder(
            latent_dim, rev_hidden_dims, input_dim, image_shape
        )
        self.latent_dim: int = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        if self.training:
            # Sample from standard normal
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, just use the mean
            return mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)
