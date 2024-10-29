import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class HamiltonianConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, kernel_3=None, kernel_4=None
    ):
        super(HamiltonianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.kernel_3 = kernel_3
        self.kernel_4 = kernel_4

        # Define nabla operator
        weights_1 = (
            torch.tensor([[2.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 2.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.weights_1 = weights_1.repeat(out_channels, 1, 1, 1)

        # Define weights for h^2/2m
        self.weights_2 = nn.Parameter(torch.randn(out_channels, 1, 3, 3))
        nn.init.orthogonal_(self.weights_2)

    def forward(self, x):
        # Combine weights to form Hamiltonian kernel
        kernel = self.weights_1 * self.weights_2
        if self.kernel_3 is not None:
            kernel = kernel + self.kernel_3
        if self.kernel_4 is not None:
            kernel = kernel + self.kernel_4

        return nn.functional.conv2d(x, kernel, padding="same")


class DIVA2D(nn.Module):
    def __init__(
        self, depth=10, filters=64, image_channels=1, kernel_size=5, use_bnorm=True
    ):
        super(DIVA2D, self).__init__()

        # Initial patches
        self.initial_patches = nn.Sequential(
            nn.Conv2d(image_channels, filters, kernel_size, padding="same"), nn.ReLU()
        )

        # Interaction layer
        self.interactions = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size, padding="same"), nn.ReLU()
        )

        # Pooling layers for kernels
        self.ori_poten_pool = nn.MaxPool2d(kernel_size=21, stride=15, padding=10)
        self.inter_kernel_pool = nn.MaxPool2d(kernel_size=21, stride=15, padding=10)

        # Hamiltonian projection
        self.proj_coef = HamiltonianConv2d(filters, filters, kernel_size)

        # Thresholding layers
        self.threshold_layers = nn.ModuleList()
        for _ in range(depth - 2):
            layer = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size, padding="same", bias=False),
                nn.BatchNorm2d(filters) if use_bnorm else nn.Identity(),
                nn.ReLU(),
            )
            self.threshold_layers.append(layer)

        # Inverse projection
        self.inv_trans = nn.Conv2d(
            filters, image_channels, kernel_size, padding="same", bias=False
        )

    def forward(self, x):
        # Initial processing
        initial = self.initial_patches(x)

        # Interaction processing
        inter = self.interactions(initial)

        # Get kernels
        ori_poten_kernel = self.ori_poten_pool(initial)
        inter_kernel = self.inter_kernel_pool(inter)

        # Project coefficients
        self.proj_coef.kernel_3 = ori_poten_kernel
        self.proj_coef.kernel_4 = inter_kernel
        out = self.proj_coef(initial)

        # Thresholding
        for layer in self.threshold_layers:
            out = layer(out)

        # Inverse projection
        out = self.inv_trans(out)

        return x - out
