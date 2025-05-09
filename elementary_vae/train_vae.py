import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict, Any, Optional

from elementary_vae.data import load_mnist
from elementary_vae.models import VAE
from elementary_vae.trainers import VAETrainer
from elementary_vae.utils import plot_reconstruction, plot_latent_space

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimension of latent space")
    parser.add_argument("--hidden-dims", type=str, default="512,256", help="Hidden dimensions, comma separated")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Weight for KL divergence term")
    parser.add_argument("--save-dir", type=str, default="./results/vae", help="Directory to save results")
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist(
        batch_size=args.batch_size,
        data_dir="./data"
    )
    print("Data loaded")
    
    # Parse hidden dimensions
    hidden_dims: List[int] = [int(dim) for dim in args.hidden_dims.split(",")]
    
    # Get image dimensions from the dataset
    sample_batch, _ = next(iter(train_loader))
    image_shape = tuple(sample_batch.shape[1:])  # (channels, height, width)
    input_dim = sample_batch[0].numel()  # total number of elements in a single image
    
    # Create model
    model = VAE(
        input_dim=input_dim,  # calculated from image dimensions
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        image_shape=image_shape
    ).to(device)
    print(f"Created VAE with architecture: {hidden_dims}, latent_dim={args.latent_dim}, input_dim={input_dim}, image_shape={image_shape}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=os.path.join(args.save_dir, "checkpoints"),
        kl_weight=args.kl_weight
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs with KL weight {args.kl_weight}")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(os.path.join(args.save_dir, "loss_plot.png"))
    
    # Get and plot reconstructions
    inputs, reconstructions = trainer.get_reconstructions(test_loader)
    fig = plot_reconstruction(inputs, reconstructions)
    fig.savefig(os.path.join(args.save_dir, "reconstructions.png"))
    
    # Plot latent space (only for 2D latent space)
    if args.latent_dim == 2:
        fig = plot_latent_space(model, test_loader, device)
        fig.savefig(os.path.join(args.save_dir, "latent_space.png"))
    else:
        print(f"Latent space visualization only available for 2D latent space (current: {args.latent_dim}D)")
    
    # Generate samples from random points in latent space
    n_samples: int = 10
    with torch.no_grad():
        z = torch.randn(n_samples, args.latent_dim).to(device)
        samples = model.decode(z)
        
    plt.figure(figsize=(n_samples, 2))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(samples[i].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "samples.png"))
    
    print(f"Training complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main() 