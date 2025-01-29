import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from diva_torch.modeling import DIVA2D
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to image matching training pattern"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def denoise_image(model, device, noisy_image, sigma):
    """Denoise a single image using the trained model"""
    # Preprocess like training data
    image_tensor = torch.from_numpy(noisy_image.astype(np.float32) / 255.0)
    image_tensor = (
        image_tensor.unsqueeze(0).unsqueeze(0).to(device)
    )  # Add batch and channel dims

    with torch.no_grad():
        model.eval()
        denoised_tensor = model(image_tensor)

    # Postprocess
    denoised_np = denoised_tensor.squeeze().cpu().numpy()
    return np.clip(denoised_np * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="DIVA2D Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save denoised image"
    )
    parser.add_argument(
        "--sigma",
        type=int,
        help="Noise level the model was trained on",
        default=15,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (adjust parameters to match your training setup)
    model = DIVA2D(depth=10, filters=64, image_channels=1, kernel_size=5)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Handle compiled models from training
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:  # Handle case where full model was saved
        model = checkpoint

    model = model.to(device)

    # Load and process image
    clean_image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    noisy_image = add_gaussian_noise(clean_image, args.sigma)
    denoised_image = denoise_image(model, device, noisy_image, args.sigma)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [clean_image, noisy_image, denoised_image]
    titles = ["Clean Image", f"Noisy (σ={args.sigma})", "Denoised Result"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    # Save results
    cv2.imwrite(args.output, denoised_image)
    comparison_path = Path(args.output).with_name(
        f"comparison_{Path(args.output).name}"
    )
    plt.savefig(str(comparison_path), bbox_inches="tight")
    plt.close()

    print(f"Saved denoised image to {args.output}")
    print(f"Saved comparison plot to {comparison_path}")


if __name__ == "__main__":
    main()
