import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class NIH(Dataset):
    """
    NIH ChestX-ray14 dataset loader (multi-label).
    - input : image (grayscale PNG)
    - label : multi-label string from df['Finding Labels'] (split by '|')

    Folder structure expected under root_path:
        root_path/
        |---- train_val/
            |---- images/          <- PNG files
            |---- descriptions/    <- train_val.csv
        |---- test/
            |---- images/
            |---- descriptions/    <- test.csv
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task = args.task

        # === Automatically set label_path and data_path based on task ===
        # If task is 'train' or 'val', use train_val split folder
        # If task is 'test', use test split folder
        if self.task in ("train", "val"):
            split_folder = "train_val"
            csv_name     = "train_val.csv"
        else:  # task == "test"
            split_folder = "test"
            csv_name     = "test.csv"

        # Construct full paths from root_path
        self.label_path = os.path.join(args.root_path, split_folder, "descriptions", csv_name)
        self.data_path  = os.path.join(args.root_path, split_folder, "images")

        print(f"[INFO] dataset    : NIH")
        print(f"[INFO] task       : {self.task}")
        print(f"[INFO] label_path : {self.label_path}")
        print(f"[INFO] data_path  : {self.data_path}")

        # === Load CSV and build image path list ===
        self.df = pd.read_csv(self.label_path)

        # Drop unnamed columns if present (artifact from some CSV exports)
        unnamed_cols = [c for c in self.df.columns if "Unnamed" in c]
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)

        # Build list of full image paths
        self.img_paths = [
            os.path.join(self.data_path, img_name)
            for img_name in self.df["Image Index"].tolist()
        ]

        print(f"[INFO] Total images loaded: {len(self.img_paths)}")

        # ===  Build transform pipeline ===
        self.fn_transform = self.get_transform()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # === Load and convert image to grayscale ===
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale: 1 channel

        if self.fn_transform is not None:
            img = self.fn_transform(img)

        # === Load label string and split by '|' for multi-label ===
        label_str = str(self.df.loc[idx, "Finding Labels"])

        return img, label_str

    def show_batch_images(self, images: torch.Tensor, labels: list):
        """
        Display all images in a given batch with their labels as titles.

        Args:
            images : Tensor of shape (B, 1, H, W) ? a full batch of images
            labels : list of label strings corresponding to each image
        """
        batch_size = images.shape[0]

        # Arrange subplots in a grid: ceil(sqrt(B)) x ceil(sqrt(B))
        cols = min(batch_size, 8)                          # max 8 columns per row
        rows = (batch_size + cols - 1) // cols             # number of rows needed

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()                    # flatten to 1D for easy indexing

        for i in range(batch_size):
            # Denormalize: reverse Normalize(mean=0.5, std=0.5) -> pixel in [0, 1]
            img_np = images[i, 0].cpu().numpy()            # shape: (H, W)
            img_np = img_np * 0.5 + 0.5                    # undo normalization
            img_np = np.clip(img_np, 0, 1)

            axes[i].imshow(img_np, cmap="gray")
            axes[i].set_title(labels[i], fontsize=7, wrap=True)
            axes[i].axis("off")

        # Hide any unused subplot slots
        for j in range(batch_size, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Batch of {batch_size} images  |  task: {self.task}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def get_transform(self):
        """
        Return a transform pipeline based on task.
        - train : includes random horizontal flip for data augmentation
        - val / test : resize and normalize only
        """
        if self.task == "train":
            return transforms.Compose([
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),   # augmentation for training
                transforms.ToTensor(),                     # (1, H, W) in [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:  # val or test
            return transforms.Compose([
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])


def parse_args():
    parser = argparse.ArgumentParser()

    # Root path that contains train_val/ and test/ folders
    parser.add_argument("-root_path",   default="/storage/hjchoi/archive/DATA")

    # Changing only -task will automatically set label_path and data_path
    parser.add_argument("-task",        default="train", choices=["train", "val", "test"])
    parser.add_argument("-image_size",  default=256, type=int)
    parser.add_argument("-image_show",  default=True, type=bool,
                        help="If True, display one batch of images after loading")
    parser.add_argument("-batch_size",  default=8, type=int)
    parser.add_argument("-num_workers", default=4, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build dataset and dataloader
    dataset = NIH(args)

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = (args.task == "train"),  # shuffle only during training
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    print(f"[INFO] Dataset size : {len(dataset)}")
    print(f"[INFO] Batches      : {len(dataloader)}")

    # Iterate one batch for verification
    for batch_id, (images, labels) in enumerate(dataloader):
        print(f"\n[Batch {batch_id}] image shape : {images.shape}")   # (B, 1, H, W)
        print(f"[Batch {batch_id}] labels      : {list(labels)}")

        # Show all images in the batch if image_show is enabled
        if args.image_show:
            dataset.show_batch_images(images, list(labels))

        if batch_id == 0:  # Only check the first batch
            break