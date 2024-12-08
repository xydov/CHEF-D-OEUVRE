import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

# Metrics functions from metrics.py
def get_ssim(img1, img2):
    # Ensure the images have the shape (B, C, H, W), where B=1, C=1 (grayscale), H=W=28
    img1 = torch.tensor(img1).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    img2 = torch.tensor(img2).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(img1, img2).item()

def get_psnr(img1, img2):
    return psnr(img1, img2, data_range=1.0)

# Load dataset
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Add random noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Autoencoder Model
def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_autoencoder()

# Train the autoencoder
checkpoint = ModelCheckpoint("denoising_model.keras", save_best_only=True, save_weights_only=False, verbose=1)
autoencoder.fit(x_train_noisy, x_train, epochs=1, batch_size=128, shuffle=True, callbacks=[checkpoint],
                validation_data=(x_test_noisy, x_test))

autoencoder = load_model("denoising_model.keras")
decoded_imgs = autoencoder.predict(x_test_noisy)

# Visualize Results with Metrics
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    # Calculate Metrics
    ssim = get_ssim(x_test[i].reshape(28, 28), decoded_imgs[i].reshape(28, 28))
    psnr_value = get_psnr(x_test[i].reshape(28, 28), decoded_imgs[i].reshape(28, 28))
    
    # Noisy Image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Degraded")
    plt.axis("off")
    
    # Denoised Image
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title(f"Reconstructed \nSSIM: {ssim:.2f}\nPSNR: {psnr_value:.2f}")
    plt.axis("off")
    
    # Original Image
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

plt.tight_layout()
plt.show()

