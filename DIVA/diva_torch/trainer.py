import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import datetime
import matplotlib.pyplot as plt
import glob
import re
import os


class DIVATrainer:
    def __init__(
        self,
        model,
        train_dataset,
        save_dir,
        batch_size=128,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize training components
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # Actual batching handled in collate_fn
            shuffle=True,
            num_workers=4,
            collate_fn=train_dataset.denoise_collate_fn,
        )

        # Loss and optimizer (matching Keras)
        self.criterion = nn.MSELoss(
            reduction="sum"
        )  # Sum reduction to match Keras sum_squared_error
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Setup logging
        self.setup_logging()

        # Training history
        self.history = {"loss": [], "epoch": []}

    def setup_logging(self):
        """Setup logging to match Keras behavior"""
        logging.basicConfig(
            filename=self.save_dir / "log.csv",
            level=logging.INFO,
            format="%(asctime)s,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def log(self, *args, **kwargs):
        """Match Keras logging format"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:")
        print(timestamp, *args, **kwargs)

    def lr_schedule(self, epoch):
        """Reproduce Keras learning rate schedule"""
        if epoch <= 20:
            lr = self.learning_rate
        elif epoch <= 30:
            lr = self.learning_rate / 10
        elif epoch <= 40:
            lr = self.learning_rate / 20
        else:
            lr = self.learning_rate / 20
        self.log(f"current learning rate is {lr:2.8f}")
        return lr

    def find_last_checkpoint(self):
        """Find the last checkpoint matching Keras functionality"""
        file_list = glob.glob(os.path.join(self.save_dir, "model_*.pth"))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(r".*model_(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch = max(epochs_exist)
        else:
            initial_epoch = 0
        return initial_epoch

    def save_checkpoint(self, epoch):
        """Save model checkpoint matching Keras format"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.save_dir / f"model_{epoch:03d}.pth")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        steps = 0

        # Update learning rate according to schedule
        lr = self.lr_schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        for batch_idx, (noisy_batches, clean_batches) in enumerate(self.train_loader):
            # Each batch contains multiple sub-batches due to the collate_fn
            for noisy_batch, clean_batch in zip(noisy_batches, clean_batches):
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(noisy_batch)

                # Calculate loss (matching Keras sum_squared_error)
                loss = (
                    self.criterion(output, clean_batch) / 2
                )  # Divide by 2 to match Keras implementation

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

                if batch_idx % 100 == 0:
                    msg = f"Train Epoch: {epoch} [{batch_idx}/{1000}] Loss: {loss.item():.6f}"
                    self.log(msg)
                    logging.info(f"{loss.item():.6f}")

                # Match Keras steps per epoch (1000)
                if steps >= 1000:
                    break
            if steps >= 1000:
                break

        avg_loss = total_loss / steps
        self.history["loss"].append(avg_loss)
        self.history["epoch"].append(epoch)
        return avg_loss

    def train(self, epochs):
        """Main training loop"""
        # Find last checkpoint (matching Keras behavior)
        initial_epoch = self.find_last_checkpoint()
        if initial_epoch > 0:
            self.log(f"resuming by loading epoch {initial_epoch:03d}")
            checkpoint = torch.load(self.save_dir / f"model_{initial_epoch:03d}.pth")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for epoch in range(initial_epoch + 1, epochs + 1):
            train_loss = self.train_epoch(epoch)

            # Save checkpoint every 10 epochs (matching Keras)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            self.log(f"Epoch: {epoch}, Loss: {train_loss:.6f}")

        # Plot training history (matching Keras visualization)
        plt.figure()
        plt.plot(self.history["epoch"], self.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train"], loc="upper right")
        plt.savefig(self.save_dir / "training_history.png")
        plt.close()
