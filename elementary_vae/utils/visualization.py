import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Union, Tuple
import matplotlib.figure


def plot_reconstruction(
    original: torch.Tensor, reconstruction: torch.Tensor, n: int = 8
) -> matplotlib.figure.Figure:
    """
    Plot original and reconstructed images side by side.

    Args:
        original: Tensor of original images
        reconstruction: Tensor of reconstructed images
        n: Number of images to plot
    """
    plt.figure(figsize=(n * 2, 4))

    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().squeeze().numpy(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstruction[i].cpu().squeeze().detach().numpy(), cmap="gray")
        plt.title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    return plt.gcf()


def plot_latent_space(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 1000,
) -> matplotlib.figure.Figure:
    """
    Plot the latent space of a VAE model.

    Args:
        model: VAE model
        dataloader: DataLoader with test data
        device: Device to run on
        n_samples: Number of samples to plot
    """
    model.eval()

    all_latents: list = []
    all_labels: list = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)

            if hasattr(model, "encode"):
                latent = model.encode(data)

                if isinstance(latent, tuple):
                    # For VAE, we get (mu, logvar), so use mu
                    latent = latent[0]
            else:
                # For regular autoencoder
                latent = model.encoder(data)

            all_latents.append(latent.cpu())
            all_labels.append(labels)

            if len(all_latents) * data.size(0) >= n_samples:
                break

    # Concatenate all latents and labels
    all_latents_tensor = torch.cat(all_latents, dim=0)[:n_samples]
    all_labels_tensor = torch.cat(all_labels, dim=0)[:n_samples]

    # If latent dim > 2, use PCA to reduce to 2D
    if all_latents_tensor.size(1) > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(all_latents_tensor.numpy())
    else:
        latent_2d = all_latents_tensor.numpy()[:, :2]

    # Plot points in 2D space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=all_labels_tensor, cmap="tab10", alpha=0.6
    )
    plt.colorbar(scatter, label="Digit class")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space Visualization")

    return plt.gcf()
