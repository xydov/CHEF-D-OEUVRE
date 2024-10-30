import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np


class DenoisingDataset(Dataset):
    def __init__(
        self,
        data_dir,
        patch_size=40,
        stride=10,
        aug_times=1,
        scales=[1],
        sigma_range=(2, 12),
        transform=None,
    ):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.aug_times = aug_times
        self.scales = scales
        self.sigma_range = sigma_range
        self.transform = transform

        # Get file list
        self.file_list = glob.glob(data_dir + "/*.png")

        # Pre-generate all patches
        self.patches = self._generate_all_patches()

    def _generate_all_patches(self):
        patches = []

        for file_name in self.file_list:
            # Read image
            clean_img = cv2.imread(file_name, 0)
            h, w = clean_img.shape

            # Generate patches for each scale
            for s in self.scales:
                h_scaled, w_scaled = int(h * s), int(w * s)

                # Extract patches
                for i in range(0, h_scaled - self.patch_size + 1, self.stride):
                    for j in range(0, w_scaled - self.patch_size + 1, self.stride):
                        clean_patch = clean_img[
                            i : i + self.patch_size, j : j + self.patch_size
                        ]

                        # Data augmentation
                        for _ in range(self.aug_times):
                            mode = np.random.randint(0, 8)
                            aug_patch = self._data_aug(clean_patch, mode)
                            patches.append(aug_patch)

        return patches

    def _data_aug(self, img, mode):
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean_patch = self.patches[idx]

        # Add noise
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.random.normal(0, sigma, clean_patch.shape)
        noisy_patch = clean_patch + noise

        # Normalize to [0,1]
        clean_patch = clean_patch.astype(np.float32) / 255.0
        noisy_patch = noisy_patch.astype(np.float32) / 255.0

        # Convert to tensor
        clean_patch = torch.from_numpy(clean_patch).unsqueeze(
            0
        )  # Add channel dimension
        noisy_patch = torch.from_numpy(noisy_patch).unsqueeze(0)

        if self.transform:
            clean_patch = self.transform(clean_patch)
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, clean_patch
