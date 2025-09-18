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
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

# UMAP for visualising latent space (install umap-learn)
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# ------------------------------------------- Datatset -------------------------------------------
"""
A PyTorch Dataset for brain MRI data.
Loads 2D MRI scans (png files), extracts the middle 2D slice, and turns it into
a normalised PyTorch tensor.
Uses nibabel to read MRI files, picks the middle slice with get_middle_slice, rescales the image,
and applies transforms (resize, normalise, etc.).
"""
class OASISSliceDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    """Returns how many MRI files in the dataset."""
    def __len__(self):
        return len(self.files)

    """
    Given an index, it loads the corresponding file (or retrieves preloaded data), extracts the
    middle slice, rescales it to [0,1], convert to [C,H,W] tensor, and ensures it's single-channel.
    This is the function that actually feeds images to your model during training.
    """
    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(f).convert("L")      # grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]

        # Add channel dimension for CNNs (C,H,W)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32)

"""
A function to grab the central axial slice from a 2D, 3D, or 4D MRI.
Returns the slice in the middle along the z-axis (like cutting the brain in half horizontally).
"""
def get_middle_slice(volume: np.ndarray) -> np.ndarray:
    """Return a 2D slice from MR volume or already-2D image."""
    if volume.ndim == 3:
        z = volume.shape[2] // 2
        return np.asarray(volume[:, :, z])
    elif volume.ndim == 4:
        z = volume.shape[2] // 2
        return np.asarray(volume[:, :, z, 0])
    elif volume.ndim == 2:
        # Already a slice
        return volume
    else:
        raise ValueError(f"Unexpected volume shape: {volume.shape}")

# --------------------------------------- VAE Model ---------------------------------------
"""
Compresses an input image into a small latent vector and then reconstructs the image from that latent vector.
- Encoder: convolutional layers shrink the image down to features.
- Latent space: two linear layers produce mean (mu) and variance (logvar).
- Reparameterization: samples a latent vector z from that distribution.
- Decoder: transpose-convolutions rebuild the image from z.
"""
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # -> (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> (128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # -> (256, 8, 8)
            nn.ReLU(),
        )

        # dynamically infer flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)  # match your slice size
            h = self.encoder(dummy)
            self.h_dim = h.numel()

        self.fc_mu = nn.Linear(self.h_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.h_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.h_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),  # output [0,1]
        )

    """
    Forward pass through the encoder.
    Returns mu and logvar for the latent distribution.
    Passes image through self.encoder, then through fc_mu and fc_logvar.
    Prepares the latent representation of the input.
    """
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

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
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)
    """
    Defines the full VAE pipeline (encode → sample → decode).
    """
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

# -------------------------------------------- Utility --------------------------------------------
"""
A helper function to save multiple tensor images into one grid image file.
Takes a batch of PyTorch images ([-1,1] range), arranges them in a grid, and saves as a single .png
(or other format).
Converts tensors to NumPy and rescales to [0,255], creates a blank grid image, pastes each image
into the grid, saves the final grid to disk.
"""
def save_image_grid(tensor_images, path, nrow=8):
    # tensor_images: (N,1,H,W) in [-1,1]
    imgs = tensor_images.detach().cpu()
    N, C, H, W = imgs.shape
    cols = nrow
    rows = int(np.ceil(N / cols))
    grid = Image.new('L', (cols * W, rows * H))
    imgs = (imgs + 1) / 2.0  # to 0..1
    imgs = (imgs * 255).numpy().astype(np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= N:
                break
            im = Image.fromarray(imgs[idx, 0])
            grid.paste(im, (c * W, r * H))
            idx += 1
    grid.save(path)

# ----------------------------------------- Training loop -----------------------------------------
"""
The main training loop for VAE.
Loads data, trains the model for several epochs, saves checkpoints, and creates visualizations
(reconstructions, latent space, UMAP).
Sets up device, data, model, optimizer, runs training loop with forward/backward passes, saves
progress (checkpoints, reconstructions, manifolds, latents).
Automates the whole VAE training process and logs progress for analysis.
"""
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Using device:', device)

    train_files = sorted(glob.glob("/home/groups/comp3710/OASIS/keras_png_slices_train/*.png"))
    val_files   = sorted(glob.glob("/home/groups/comp3710/OASIS/keras_png_slices_validate/*.png"))
    test_files  = sorted(glob.glob("/home/groups/comp3710/OASIS/keras_png_slices_test/*.png"))

    ds = OASISSliceDataset(train_files)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        for batch_idx, data in enumerate(dl):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kld = vae_loss(recon, data, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dl)}]  loss={loss.item():.1f} recon={recon_loss.item():.1f} kld={kld.item():.1f}")

        print(f"Epoch {epoch} summary: loss={epoch_loss:.1f} recon={epoch_recon:.1f} kld={epoch_kld:.1f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_epoch_{epoch}.pth'))

        # Save some reconstructions from validation (use first batch)
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dl))[:args.recon_n]
            sample_batch = sample_batch.to(device)
            recon, mu, logvar = model(sample_batch)
            cat = torch.cat([sample_batch, recon], dim=0)  # originals then recons
            save_image_grid(cat, os.path.join(args.out_dir, f'recons_epoch_{epoch}.png'), nrow=args.recon_n)

        # If latent_dim==2, save manifold grid by decoding a grid of z
        if args.latent_dim == 2:
            n = args.manifold_n
            # create grid in latent space between -3 and 3
            grid_x = np.linspace(-args.manifold_range, args.manifold_range, n)
            grid_y = np.linspace(-args.manifold_range, args.manifold_range, n)
            zs = []
            for yi in grid_y[::-1]:
                for xi in grid_x:
                    zs.append([xi, yi])
            z = torch.tensor(zs, dtype=torch.float32).to(device)
            with torch.no_grad():
                imgs = model.decode(z).cpu()
                save_image_grid(imgs, os.path.join(args.out_dir, f'manifold_epoch_{epoch}.png'), nrow=n)

        # Save latent embeddings for UMAP
        if args.save_latents:
            all_mu = []
            for data in dl:
                data = data.to(device)
                with torch.no_grad():
                    mu, logvar = model.encode(data)
                all_mu.append(mu.cpu().numpy())
            all_mu = np.concatenate(all_mu, axis=0)
            np.save(os.path.join(args.out_dir, f'latents_epoch_{epoch}.npy'), all_mu)

            if _HAS_UMAP and args.umap:
                reducer = umap.UMAP(n_components=2)
                emb = reducer.fit_transform(all_mu)
                plt.figure(figsize=(6,6))
                plt.scatter(emb[:,0], emb[:,1], s=4)
                plt.title(f"UMAP of latents (epoch {epoch})")
                plt.savefig(os.path.join(args.out_dir, f'umap_epoch_{epoch}.png'))
                plt.close()

    print('Training finished')