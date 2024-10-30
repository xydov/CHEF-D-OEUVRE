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
        sigma=15,
        num_noise_realiza=2,
    ):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.aug_times = aug_times
        self.scales = scales
        self.sigma = sigma
        self.num_noise_realiza = num_noise_realiza

        # Get file list
        self.file_list = glob.glob(data_dir + "/*.png")

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

    def _noise_aug(self, mode):
        """Matches Keras discrete noise levels"""
        noise_levels = {
            0: 2,
            1: 4,
            2: 6,
            3: 8,
            4: 10,
            5: 12,
            6: 14,
            7: 16,
            8: 18,
            9: 20,
        }
        return noise_levels.get(mode, self.sigma)

    def __len__(self):
        return len(self.file_list) * self.num_noise_realiza

    def __getitem__(self, idx):
        file_idx = idx // self.num_noise_realiza
        file_name = self.file_list[file_idx]

        # Read image
        clean_img = cv2.imread(file_name, 0)
        h, w = clean_img.shape

        patches = []
        clean_patches = []

        # Get noise level (either fixed or augmented)
        sigma = self._noise_aug(mode=np.random.randint(0, 6))

        for s in self.scales:
            h_scaled, w_scaled = int(h * s), int(w * s)

            # Extract patches
            for i in range(0, h_scaled - self.patch_size + 1, self.stride):
                for j in range(0, w_scaled - self.patch_size + 1, self.stride):
                    clean_x = clean_img[
                        i : i + self.patch_size, j : j + self.patch_size
                    ]

                    # Data augmentation with different rotation
                    for k in range(self.aug_times):
                        mode_k = np.random.randint(0, 8)
                        clean_x_aug = self._data_aug(clean_x, mode=mode_k)

                        # Add noise
                        noise = np.random.normal(0, sigma, clean_x_aug.shape)
                        noisy_x_aug = clean_x_aug + noise

                        patches.append(noisy_x_aug)
                        clean_patches.append(clean_x_aug)

        return patches, clean_patches


def denoise_collate_fn(batch):
    """Custom collate function for the denoising dataset"""
    noisy_patches = []
    clean_patches = []

    # Unpack the batch
    for noisy_list, clean_list in batch:
        noisy_patches.extend(noisy_list)
        clean_patches.extend(clean_list)

    # Convert to numpy arrays
    noisy_patches = np.array(noisy_patches, dtype=np.float32) / 255.0
    clean_patches = np.array(clean_patches, dtype=np.float32) / 255.0

    # Convert to tensors and add channel dimension
    noisy_patches = torch.from_numpy(noisy_patches).unsqueeze(1)
    clean_patches = torch.from_numpy(clean_patches).unsqueeze(1)

    # Ensure we have complete batches
    batch_size = 128  # You might want to make this configurable
    num_complete_batches = len(noisy_patches) // batch_size
    if num_complete_batches == 0:
        raise RuntimeError("Not enough patches to form a complete batch")

    # Only keep complete batches
    noisy_patches = noisy_patches[: num_complete_batches * batch_size]
    clean_patches = clean_patches[: num_complete_batches * batch_size]

    # Reshape into batches
    noisy_patches = noisy_patches.view(
        num_complete_batches,
        batch_size,
        1,
        noisy_patches.size(-2),
        noisy_patches.size(-1),
    )
    clean_patches = clean_patches.view(
        num_complete_batches,
        batch_size,
        1,
        clean_patches.size(-2),
        clean_patches.size(-1),
    )

    return noisy_patches, clean_patches
