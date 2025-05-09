import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
from typing import Optional, Tuple, List


class VAETrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        kl_weight: float = 1.0,
        use_kl_annealing: bool = True,
        kl_anneal_cycles: int = 1,
    ) -> None:
        super().__init__(model, optimizer, device, checkpoint_dir)
        self.max_kl_weight: float = kl_weight
        self.current_kl_weight: float = 0.0 if use_kl_annealing else kl_weight
        self.use_kl_annealing: bool = use_kl_annealing
        self.kl_anneal_cycles: int = kl_anneal_cycles
        self.current_epoch: int = 0
        self.num_epochs: int = 0

    def _update_kl_weight(self) -> None:
        """Update KL weight based on annealing schedule"""
        if not self.use_kl_annealing:
            self.current_kl_weight = self.max_kl_weight
            return

        # Cyclical annealing schedule
        cycle_size = max(1, self.num_epochs // self.kl_anneal_cycles)
        cycle_position = self.current_epoch % cycle_size
        normalized_position = cycle_position / cycle_size

        # Linear schedule to smoothly transition from 0 to max_kl_weight
        self.current_kl_weight = min(
            normalized_position * 2.0 * self.max_kl_weight, self.max_kl_weight
        )

        print(
            f"Epoch {self.current_epoch+1}/{self.num_epochs} - KL weight: {self.current_kl_weight:.6f}"
        )

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

        self.num_epochs = num_epochs

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Update KL weight for this epoch
            self._update_kl_weight()

            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

        # Save the final model
        self.save_checkpoint("final_model.pt")
        return train_losses, val_losses

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss: float = 0.0
        total_recon_loss: float = 0.0
        total_kl_loss: float = 0.0
        total_weighted_kl_loss: float = 0.0

        for data, _ in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            reconstruction, mu, logvar = self.model(data)

            # Flatten the inputs and reconstructions for binary cross entropy
            flattened_data = data.view(data.size(0), -1)
            flattened_recon = reconstruction.view(reconstruction.size(0), -1)

            # Calculate reconstruction loss (per pixel)
            recon_loss = F.binary_cross_entropy(
                flattened_recon, flattened_data, reduction="sum"
            ) / data.size(0)

            # Calculate KL divergence (per latent dimension)
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            )
            weighted_kl_loss = self.current_kl_weight * kl_loss

            # Total loss - use the current KL weight
            loss = recon_loss + weighted_kl_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_weighted_kl_loss += weighted_kl_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_weighted_kl_loss = total_weighted_kl_loss / len(train_loader)

        print(
            f"Train - Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Weighted KL: {avg_weighted_kl_loss:.4f}"
        )

        return avg_loss

    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss: float = 0.0
        total_recon_loss: float = 0.0
        total_kl_loss: float = 0.0
        total_weighted_kl_loss: float = 0.0

        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validating", leave=False):
                data = data.to(self.device)

                # Forward pass
                reconstruction, mu, logvar = self.model(data)

                # Flatten the inputs and reconstructions for binary cross entropy
                flattened_data = data.view(data.size(0), -1)
                flattened_recon = reconstruction.view(reconstruction.size(0), -1)

                # Calculate reconstruction loss
                recon_loss = F.binary_cross_entropy(
                    flattened_recon, flattened_data, reduction="sum"
                ) / data.size(0)

                # Calculate KL divergence
                kl_loss = (
                    -0.5
                    * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    / data.size(0)
                )
                weighted_kl_loss = self.current_kl_weight * kl_loss

                # Total loss - use the current KL weight to be consistent with training
                loss = recon_loss + weighted_kl_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_weighted_kl_loss += weighted_kl_loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        avg_weighted_kl_loss = total_weighted_kl_loss / len(val_loader)

        print(
            f"Val - Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Weighted KL: {avg_weighted_kl_loss:.4f}"
        )

        return avg_loss
