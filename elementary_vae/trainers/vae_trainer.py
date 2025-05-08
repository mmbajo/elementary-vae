import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_trainer import BaseTrainer
from typing import Optional

class VAETrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        kl_weight: float = 1.0
    ) -> None:
        super().__init__(model, optimizer, device, checkpoint_dir)
        self.kl_weight: float = kl_weight
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss: float = 0.0
        total_recon_loss: float = 0.0
        total_kl_loss: float = 0.0
        
        for data, _ in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = self.model(data)
            
            # Calculate reconstruction loss
            recon_loss = F.binary_cross_entropy(reconstruction, data)
            
            # Calculate KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / data.size(0)  # Normalize by batch size
            
            # Total loss
            loss = recon_loss + self.kl_weight * kl_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        print(f"Train - Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss: float = 0.0
        total_recon_loss: float = 0.0
        total_kl_loss: float = 0.0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validating", leave=False):
                data = data.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(data)
                
                # Calculate reconstruction loss
                recon_loss = F.binary_cross_entropy(reconstruction, data)
                
                # Calculate KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / data.size(0)  # Normalize by batch size
                
                # Total loss
                loss = recon_loss + self.kl_weight * kl_loss
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        
        print(f"Val - Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        return avg_loss 