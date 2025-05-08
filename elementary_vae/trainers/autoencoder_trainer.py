import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_trainer import BaseTrainer

class AutoencoderTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        device,
        checkpoint_dir="./checkpoints",
    ):
        super().__init__(model, optimizer, device, checkpoint_dir)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
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
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validating", leave=False):
                data = data.to(self.device)
                
                # Forward pass
                reconstruction = self.model(data)
                
                # Calculate reconstruction loss
                loss = F.binary_cross_entropy(reconstruction, data)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader) 