import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class Encoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Implement the encoder layers here
        # You should create layers to process input data
        # and project it to the latent space
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to flatten the input image first
        o = x.view(x.size(0), -1)
        o = F.relu(self.fc1(o))
        o = F.relu(self.fc2(o))
        o = self.fc3(o)
        return o

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dims: List[int] = [256, 512], output_dim: int = 784) -> None:
        super().__init__()
        
        # TODO: Implement the decoder layers here
        # You should create layers to process the latent vector
        # and reconstruct the original input
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Remember to reshape the output to match the original image dimensions
        o = F.relu(self.fc1(z))
        o = F.relu(self.fc2(o))
        o = torch.sigmoid(self.fc3(o))
        return o.view(-1, 1, 28, 28) # hardcoded for MNIST

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [512, 256], latent_dim: int = 32) -> None:
        super().__init__()
        
        # TODO: Create the encoder and decoder components
        rev_hidden_dims = hidden_dims[::-1]
        self.encoder: Optional[Encoder] = Encoder(input_dim, hidden_dims, latent_dim)  # Replace with your encoder
        self.decoder: Optional[Decoder] = Decoder(latent_dim, rev_hidden_dims, input_dim)  # Replace with your decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the full autoencoder forward pass
        # 1. Encode the input
        # 2. Decode the latent representation
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        
        return reconstructed  # Replace with the reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement encoding only
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement decoding only
        return self.decoder(z)