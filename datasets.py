import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import os

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class MyDataset(Dataset):
    def __init__(self, data_path, img_size=416, augment=True):
        self.root_path = '/'.join(data_path.split('/')[:-1])
        self.augment = augment
        self.img_size = img_size

        with open(data_path, "r") as f:
            self.img_files = f.readlines()

        self.img_num = len(self.img_files)

        self.label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files]

    def __getitem__(self, index):
        img_path = self.root_path + self.img_files[index % self.img_num].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        label_path = self.root_path + self.label_files[index % self.img_num].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)

            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w / padded_w
            boxes[:, 4] *= h / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)

            #print("111: ", targets)
            return img, targets

    def __len__(self):
        return self.img_num

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        targets = [boxes for boxes in targets if boxes is not None]

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        return imgs, targets


if __name__ == '__main__':
    import time

    #dataset = MyDataset(data_path='D:/datasets/coco/trainvalno5k.txt')
    dataset = MyDataset(data_path='D:/datasets/voc-custom/train.txt')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    for batch_i, (imgs, targets) in enumerate(dataloader):

        print(imgs.shape, targets.shape)

        time.sleep(60)