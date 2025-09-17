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