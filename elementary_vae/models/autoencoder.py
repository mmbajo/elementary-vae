import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=32):
        super().__init__()
        
        # TODO: Implement the encoder layers here
        # You should create layers to process input data
        # and project it to the latent space
        
    def forward(self, x):
        # TODO: Implement the forward pass
        # Remember to flatten the input image first
        
        return None  # Replace with your latent representation

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=[256, 512], output_dim=784):
        super().__init__()
        
        # TODO: Implement the decoder layers here
        # You should create layers to process the latent vector
        # and reconstruct the original input
        
    def forward(self, z):
        # TODO: Implement the forward pass
        # Remember to reshape the output to match the original image dimensions
        
        return None  # Replace with your reconstruction

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=32):
        super().__init__()
        
        # TODO: Create the encoder and decoder components
        self.encoder = None  # Replace with your encoder
        self.decoder = None  # Replace with your decoder
        
    def forward(self, x):
        # TODO: Implement the full autoencoder forward pass
        # 1. Encode the input
        # 2. Decode the latent representation
        
        return None  # Replace with the reconstruction
    
    def encode(self, x):
        # TODO: Implement encoding only
        return None
    
    def decode(self, z):
        # TODO: Implement decoding only
        return None 