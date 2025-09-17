"""
VAE training script for Preprocessed OASIS MR brain images on Rangpur cluster.

Features:
- Loads NIfTI (.nii, .nii.gz) files from data directory, extracts the central axial slice by default.
- Builds a convolutional VAE (encoder/decoder) in PyTorch.
- Supports GPU training (CUDA) if available.
- Saves checkpoints, reconstructed images, a latent-space sample grid (if latent_dim==2),
and a UMAP embedding of the latent vectors.
"""