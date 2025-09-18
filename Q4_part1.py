"""
VAE training script for Preprocessed OASIS MR brain images on Rangpur cluster.

Features:
- Loads NIfTI (.nii, .nii.gz) files from data directory, extracts the central axial slice by default.
- Builds a convolutional VAE (encoder/decoder) in PyTorch.
- Supports GPU training (CUDA) if available.
- Saves checkpoints, reconstructed images, a latent-space sample grid (if latent_dim==2),
and a UMAP embedding of the latent vectors.
"""

import os
import glob

import numpy as np
import nibabel as nib

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# UMAP for visualising latent space (install umap-learn)
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# ------------------------------------------- Datatset -------------------------------------------
"""
A PyTorch Dataset for brain MRI data.
Loads 3D MRI scans (NIfTI files, .nii or .nii.gz), extracts the middle 2D slice, and turns it into
a normalized PyTorch tensor.
Uses nibabel to read MRI files, picks the middle slice with get_middle_slice, rescales the image,
and applies transforms (resize, normalize, etc.).
"""
class OASISSliceDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 128, preload=False):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii*"), recursive=True))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .nii/.nii.gz files found in {root_dir}")
        self.img_size = img_size
        self.preload = preload
        self.data = []
        if preload:
            for f in self.files:
                vol = nib.load(f).get_fdata()
                sl = get_middle_slice(vol)
                self.data.append(sl.astype(np.float32))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # -> [C,H,W] with values 0-1
            transforms.Normalize((0.5,), (0.5,)), # scale to [-1,1]
        ])

    """Returns how many MRI files in the dataset."""
    def __len__(self):
        return len(self.files)

    """
    Given an index, it loads the corresponding file (or retrieves preloaded data), extracts the
    middle slice, rescales it to [0,1], convert to [C,H,W] tensor, and ensures it's single-channel.
    This is the function that actually feeds images to your model during training.
    """
    def __getitem__(self, idx):
        if self.preload:
            img = self.data[idx]
        else:
            f = self.files[idx]
            vol = nib.load(f).get_fdata()
            img = get_middle_slice(vol)
        # img is 2D numpy float
        img = np.nan_to_num(img, nan=0.0)
        # normalize per-image to 0..1
        mi = img.min()
        ma = img.max()
        if ma - mi > 0:
            img = (img - mi) / (ma - mi)
        else:
            img = img - mi
        img = (img * 255).astype(np.uint8)
        tensor = self.transform(img)
        # ensure single channel
        if tensor.shape[0] == 3:
            tensor = tensor.mean(dim=0, keepdim=True)
        return tensor

"""
A function to grab the central axial slice from a 3D MRI.
Returns the slice in the middle along the z-axis (like cutting the brain in half horizontally).
"""
def get_middle_slice(volume: np.ndarray) -> np.ndarray:
    """Return central axial slice (axis 2 assumed as axial)."""
    # Try common axes ordering
    if volume.ndim == 3:
        z = volume.shape[2] // 2
        return np.asarray(volume[:, :, z])
    elif volume.ndim == 4:
        # sometimes volumes have singleton 4th dim
        z = volume.shape[2] // 2
        return np.asarray(volume[:, :, z, 0])
    else:
        raise ValueError("Unexpected volume shape: {}".format(volume.shape))

# --------------------------------------- VAE Model ---------------------------------------
"""
Compresses an input image into a small latent vector and then reconstructs the image from that latent vector.
- Encoder: convolutional layers shrink the image down to features.
- Latent space: two linear layers produce mean (mu) and variance (logvar).
- Reparameterization: samples a latent vector z from that distribution.
- Decoder: transpose-convolutions rebuild the image from z.
"""
class ConvVAE(nn.Module):
    def __init__(self, img_channels=1, feature_dim=64, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            # input: (B,1,128,128)
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),  # 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),  # 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),  # 8x8
            nn.Flatten(),
        )
        # compute flattened size dynamically by passing a dummy
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, 128, 128)
            h = self.encoder(dummy)
            flat_dim = h.shape[1]
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),  # 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),  # 64x64
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Tanh(),  # outputs in [-1,1]
        )

    """
    Forward pass through the encoder.
    Returns mu and logvar for the latent distribution.
    Passes image through self.encoder, then through fc_mu and fc_logvar.
    Prepares the latent representation of the input.
    """
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    """
    The "sampling trick" of VAEs.
    Samples a latent vector z ~ N(mu, \theta^2).
    Compute std = exp(0.5*logvar), sample noise eps from standard normal, return mu + eps*std.
    Makes sampling differentiable, so gradients can flow during training.
    """
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    """
    Forward pass through the decoder.
    Turns a latent vector into a reconstructed image.
    Linear layer expands z, then transpose convolutions grow it back into (128x128).
    Generates new data (images) from the latent code.
    """
    def decode(self, z):
        h = self.fc_dec(z)
        x = self.decoder(h)
        return x

    """
    Defines the full VAE pipeline (encode → sample → decode).
    """
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xrec = self.decode(z)
        return xrec, mu, logvar

# --------------------------------------------- Loss ---------------------------------------------
"""
The loss function used to train a Variational Autoencoder (VAE).
Combines two losses:
- Reconstruction loss: how close the output image is to the input.
- KL divergence loss: how close the latent space distribution is to a standard normal (N(0,1)).
Uses mean squared error (MSE) for reconstruction, and a closed-form formula for KL divergence.
VAEs need both losses to balance accurate reconstructions and a well-behaved latent space.
"""
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # recon_x and x are in [-1,1] (Tanh). Use MSE reconstruction loss.
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence between N(mu,var) and N(0,1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld