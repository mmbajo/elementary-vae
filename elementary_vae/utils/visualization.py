import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    plot_3d: bool = True,
) -> matplotlib.figure.Figure:
    """
    Plot the latent space of a VAE model.
    For 3D plotting:
    - Uses PCA if the latent dimension is > 3
    - Uses the original dimensions if latent_dim = 3
    - Pads with zeros if latent_dim < 3
    
    For 2D plotting:
    - Uses PCA if the latent dimension is > 2
    - Uses the original dimensions if latent_dim = 2
    - Pads with zeros if latent_dim = 1

    Args:
        model: VAE model
        dataloader: DataLoader with test data
        device: Device to run on
        n_samples: Number of samples to plot
        plot_3d: Whether to use 3D plotting (True) or 2D plotting (False)
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

    # Check latent dimension
    latent_dim = all_latents_tensor.size(1)
    
    if plot_3d:
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine how to handle the latent space based on its dimension
        if latent_dim > 3:
            # Reduce to 3D using PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            latent_3d = pca.fit_transform(all_latents_tensor.numpy())
            
            # Get explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            total_explained_variance = np.sum(explained_variance)
            
            # Set title
            title = f"PCA-Reduced Latent Space ({latent_dim}D → 3D)"
            fig.suptitle(title)
            
            # Add variance information
            fig.text(
                0.5, 
                0.01, 
                f"Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}, PC3={explained_variance[2]:.2%}, Total={total_explained_variance:.2%}",
                ha="center", 
                fontsize=10
            )
        elif latent_dim == 3:
            # Use all 3 dimensions directly
            latent_3d = all_latents_tensor.numpy()
            title = "3D Latent Space"
            fig.suptitle(title)
        else:
            # Pad with zeros if latent_dim < 3
            latent_3d = np.zeros((all_latents_tensor.shape[0], 3))
            latent_3d[:, :latent_dim] = all_latents_tensor.numpy()
            title = f"{latent_dim}D Latent Space (padded to 3D)"
            fig.suptitle(title)
        
        # Create the 3D scatter plot
        scatter = ax.scatter(
            latent_3d[:, 0], 
            latent_3d[:, 1], 
            latent_3d[:, 2],
            c=all_labels_tensor, 
            cmap="tab10", 
            alpha=0.8,
            s=40,  # Marker size
            edgecolors='w',  # White edges for better visibility
            linewidth=0.5
        )
        
        # Add labels and colorbar
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.colorbar(scatter, ax=ax, label="Digit class", pad=0.1)
        
        # Improve 3D visualization
        ax.view_init(elev=30, azim=45)  # Set initial view angle
        ax.dist = 11  # Adjust camera distance
        
        # Better layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title and caption
        
        return fig
    else:
        # Create figure for 2D plotting (original behavior)
        plt.figure(figsize=(10, 8))
        
        title_prefix = "Latent Space Visualization"
        
        # If latent dim > 2, use PCA to reduce to 2D
        if latent_dim > 2:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(all_latents_tensor.numpy())
            
            # Get explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            total_explained_variance = np.sum(explained_variance)
            
            # Update title and add variance information
            title_prefix = f"PCA-Reduced Latent Space ({latent_dim}D → 2D)"
            plt.figtext(
                0.5, 
                0.01, 
                f"Explained variance: PC1 = {explained_variance[0]:.2%}, PC2 = {explained_variance[1]:.2%}, Total = {total_explained_variance:.2%}",
                ha="center", 
                fontsize=10
            )
        elif latent_dim == 2:
            latent_2d = all_latents_tensor.numpy()
        else:  # latent_dim == 1
            # Pad with zeros for the second dimension
            latent_2d = np.zeros((all_latents_tensor.shape[0], 2))
            latent_2d[:, 0] = all_latents_tensor.numpy().squeeze()
            title_prefix = "1D Latent Space (padded to 2D)"

        # Plot points in 2D space
        scatter = plt.scatter(
            latent_2d[:, 0], latent_2d[:, 1], c=all_labels_tensor, cmap="tab10", alpha=0.6
        )
        plt.colorbar(scatter, label="Digit class")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title(title_prefix)
        plt.tight_layout()

        return plt.gcf()
