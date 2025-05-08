import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_trainer import BaseTrainer
from typing import Optional

class AutoencoderTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        super().__init__(model, optimizer, device, checkpoint_dir)
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss: float = 0.0
        
        for data, _ in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstruction = self.model(data)
            
            # Calculate reconstruction loss (MSE or BCE)
            loss = F.binary_cross_entropy(reconstruction, data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss: float = 0.0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validating", leave=False):
                data = data.to(self.device)
                
                # Forward pass
                reconstruction = self.model(data)
                
                # Calculate reconstruction loss
                loss = F.binary_cross_entropy(reconstruction, data)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader) 