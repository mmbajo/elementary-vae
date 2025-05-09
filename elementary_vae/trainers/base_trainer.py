import os
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Union


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Abstract method to train for one epoch"""
        raise NotImplementedError

    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Abstract method to validate the model"""
        raise NotImplementedError

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        save_interval: int = 5,
    ) -> Tuple[List[float], List[float]]:
        """Train the model for the specified number of epochs"""
        best_val_loss = float("inf")
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(num_epochs):
            # Training step
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validation step
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")

            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

        # Save the final model
        self.save_checkpoint("final_model.pt")
        return train_losses, val_losses

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")

    def get_reconstructions(
        self, dataloader: torch.utils.data.DataLoader, num_samples: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate reconstructions for a batch of inputs"""
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(dataloader)
            inputs = next(data_iter)[0][:num_samples].to(self.device)
            outputs = self.model(inputs)

            if isinstance(outputs, tuple):
                # For VAE which returns (reconstruction, mu, logvar)
                reconstructions = outputs[0]
            else:
                # For Autoencoder which just returns reconstruction
                reconstructions = outputs

            return inputs, reconstructions
