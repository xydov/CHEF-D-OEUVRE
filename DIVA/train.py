from diva_torch.trainer import DIVATrainer
from diva_torch.dataset import DenoisingDataset
from diva_torch.modeling import DIVA2D
import argparse


def main():
    parser = argparse.ArgumentParser(description="PyTorch DIVA2D")
    parser.add_argument(
        "--model", default="DIVA2D", type=str, help="choose a type of model"
    )
    parser.add_argument("--kernel_size", default=5, type=int, help="kernel size")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--train_data",
        default="./data/training_set",
        type=str,
        help="path of train data",
    )
    parser.add_argument("--sigma", default=15, type=int, help="noise level")
    parser.add_argument(
        "--epoch", default=100, type=int, help="number of train epoches"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="initial learning rate for Adam"
    )
    args = parser.parse_args()

    # Create save directory
    save_dir = f"./models/{args.model}_sigma_{args.sigma}"

    # Create dataset
    train_dataset = DenoisingDataset(
        args.train_data, sigma=args.sigma, num_noise_realiza=2
    )

    # Create model
    model = DIVA2D(depth=10, filters=64, image_channels=1, kernel_size=args.kernel_size)

    # Create trainer
    trainer = DIVATrainer(
        model=model,
        train_dataset=train_dataset,
        save_dir=save_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Train model
    trainer.train(args.epoch)


if __name__ == "__main__":
    main()
