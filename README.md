# Elementary VAE

A modular implementation of autoencoders and variational autoencoders (VAEs) for practice, with the MNIST dataset.

## Setup

This project uses `uv` for package management.

### Installing uv

You can install uv using one of the following methods:

#### Option 1: Using the installation script

```bash
# Run the installation script which will check if uv is installed
# and install it if necessary
./install.sh
```

#### Option 2: Manual installation

```bash
# Install Rust (required for uv)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Install uv using cargo
cargo install uv

# Install dependencies
uv pip install -e .
```

### Running the experiments

```bash
# Run a simple autoencoder experiment
python -m elementary_vae.train_autoencoder

# Run a VAE experiment
python -m elementary_vae.train_vae
```

#### Command-line arguments

Both training scripts support the following arguments:

```
--batch-size     Batch size for training (default: 128)
--epochs         Number of training epochs (default: 20)
--lr             Learning rate (default: 1e-3)
--latent-dim     Dimension of latent space (default: 32)
--hidden-dims    Hidden dimensions, comma separated (default: "512,256")
--save-dir       Directory to save results
```

For the VAE script, there's an additional parameter:

```
--kl-weight      Weight for the KL divergence term (default: 1.0)
```

## Project Structure

- `elementary_vae/`: Main package
  - `data/`: Dataset loading and processing
  - `models/`: Neural network model definitions
  - `trainers/`: Training utilities
  - `utils/`: Helper functions and utilities
  - `train_autoencoder.py`: Script to train an autoencoder
  - `train_vae.py`: Script to train a variational autoencoder
