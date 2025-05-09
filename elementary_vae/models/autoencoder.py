import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        # Create a dynamic list of layers based on hidden_dims
        layers = []
        input_size = input_dim

        # Add all hidden layers
        for hidden_size in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Add final layer to latent space
        layers.append(nn.Linear(input_size, latent_dim))

        # Create sequential model with all layers
        self.encoder_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Pass through all layers
        return self.encoder_net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: List[int] = [256, 512],
        output_dim: int = 784,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        # Create a dynamic list of layers based on hidden_dims
        layers = []
        input_size = latent_dim

        # Add all hidden layers
        for hidden_size in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Add final output layer with sigmoid activation
        layers.append(nn.Linear(input_size, output_dim))
        layers.append(nn.Sigmoid())

        # Create sequential model with all layers
        self.decoder_net = nn.Sequential(*layers)

        # Store output shape for reshaping in forward pass
        self.output_dim = output_dim
        self.image_shape = image_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Pass through all layers
        output = self.decoder_net(z)
        # Reshape output to match original image dimensions
        return output.view(output.size(0), *self.image_shape)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 32,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        # Use the same hidden dimensions structure for encoder and decoder (reversed)
        rev_hidden_dims = hidden_dims[::-1]
        self.encoder: Optional[Encoder] = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder: Optional[Decoder] = Decoder(
            latent_dim, rev_hidden_dims, input_dim, image_shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
