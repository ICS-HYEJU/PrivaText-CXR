# import argparse
# from mpmath.identification import transforms
# from torch.utils.data import Dataset
# import numpy as np
# from scipy import io
# import torch
# import torchvision.transforms as transforms
# import glob
# import pandas as pd
# from PIL import Image
# import os
#
# class NIH(Dataset):
#     def __init__(self, args):
#         super().__init__()
#         self.label = pd.read_csv(args.label_path)
#         self.labelPath = self.label['Image Index']
#         self.img_path = []
#         for path in self.labelPath:
#             self.img_path.append(os.path.join(args.data_path, path))
#         if args.image_show:
#             from PIL import Image
#             import matplotlib.pyplot as plt
#
#             img = Image.open(self.img_path[0]).convert("L")  # grayscale
#             img = np.array(img)
#
#             print(img.shape, img.dtype)  # (H, W), uint8
#
#             plt.imshow(img, cmap="gray")
#             plt.axis("off")
#             plt.show()
#
#         self.fn_transform = self.get_transform()
#         self.task = args.task
#         #
#         self.df = pd.read_csv(args.label_path)
#
#     def __len__(self):
#         return len(self.path)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.img_path[idx])
#
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-data_path', default= '/storage/hjchoi/archive/image_file/images')
#     parser.add_argument('-label_path', default='/storage/hjchoi/archive/Data_Entry_2017.csv')
#     parser.add_argument('-task', default= 'train', help=['train/val/test'])
#     parser.add_argument('-image_size', default=512, type=int)
#     parser.add_argument('-image_show', default=True, type=bool, help='If you want to show the image, True')
#     args = parser.parse_args()
#
#     CXR = NIH(args)
#

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms


class NIH(Dataset):
    """
    NIH ChestX-ray14 dataset loader (multi-label).
    - input : image
    - label : multi-label from df['Finding Labels'] (split by '|')
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task = args.task
        self.df = pd.read_csv(args.label_path).drop('Unnamed: 11',axis=1)
        self.img_paths = [
            os.path.join(args.data_path, img_name)
            for img_name in self.df["Image Index"].tolist()
        ]
        self.fn_transform = self.get_transform()

        if args.image_show:
            self._show_first_image()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ----- image -----
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale

        if self.fn_transform is not None:
            img = self.fn_transform(img)

        # ----- label -----
        label_str = str(self.df.loc[idx, "Finding Labels"])
        labels = [x.strip() for x in label_str.split("|")]

        return img, label_str

    def _show_first_image(self,idx=0):
        import matplotlib.pyplot as plt

        img = Image.open(self.img_paths[idx]).convert("L")
        arr = np.array(img)
        print("[DEBUG] first image:", arr.shape, arr.dtype)

        plt.imshow(arr, cmap="gray")
        plt.axis("off")
        plt.show()

        print("[DEBUG] first label string:", self.df.loc[0, "Finding Labels"])

    def get_transform(self):
        if self.task == "train":
            return transforms.Compose([
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # (1,H,W) in [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default="/storage/hjchoi/archive/image_file")
    parser.add_argument("-label_path", default="/storage/hjchoi/archive/Data_Entry_2017.csv")
    parser.add_argument("-task", default="train", choices=["train", "val", "test"])
    parser.add_argument("-image_size", default=256, type=int)
    parser.add_argument("-image_show", default=True, help="show_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = NIH(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
    )
    for batch_id, data in enumerate(dataloader):
        if batch_id == 1:
            break
        image, label = data[0], data[1]
        print(label)