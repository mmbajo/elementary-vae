"""Data loading and processing utilities."""

from .mnist_dataset import load_mnist
from typing import List

__all__: List[str] = ["load_mnist"] 