"""
Dataset and DataLoader utilities for BDD100K
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (320, 640)


class BDD100KDataset(Dataset):
    """BDD100K dataset for multi-task learning.

    Args:
        root_dir: path to BDD100K image root (contains images/)
        split: 'train' or 'val'
        tasks: list of tasks to load, e.g. ['seg', 'depth', 'lanes']
        seg_root: separate root for segmentation masks (contains train/, val/)
        depth_root: separate root for depth maps (contains train/, val/, test/)
        transform: optional additional transforms
    """

    def __init__(self, root_dir, split='train', tasks=None,
                 seg_root=None, depth_root=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.tasks = tasks or ['seg', 'depth']
        self.transform = transform

        # Image directory — try 10k first, then 100k
        self.img_dir = os.path.join(root_dir, 'images', '10k', split)
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(root_dir, 'images', '100k', split)

        # Segmentation mask directory
        if seg_root:
            self.seg_dir = os.path.join(seg_root, split)
        else:
            self.seg_dir = os.path.join(root_dir, 'labels', 'sem_seg', 'masks', split)

        # Depth — search all splits since depth dataset may split differently
        self.depth_root = depth_root or os.path.join(root_dir, 'labels', 'depth')
        self.depth_splits = ['train', 'val', 'test']

        # Lane segmentation directory
        self.lane_dir = os.path.join(root_dir, 'labels', 'lane', 'masks', split)

        # Collect image filenames — only keep images that have ALL requested labels
        all_filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        self.filenames = []
        for f in all_filenames:
            basename = os.path.splitext(f)[0]
            keep = True
            if 'seg' in self.tasks:
                if not os.path.exists(os.path.join(self.seg_dir, basename + '_train_id.png')):
                    keep = False
            if 'depth' in self.tasks:
                if not self._find_depth(basename):
                    keep = False
            if keep:
                self.filenames.append(f)

        # Standard image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _find_depth(self, basename):
        """Search all depth splits for this image."""
        for split in self.depth_splits:
            p = os.path.join(self.depth_root, split, basename + '.jpg')
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        basename = os.path.splitext(filename)[0]

        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        sample = {'image': image, 'filename': basename}

        if 'seg' in self.tasks:
            seg_path = os.path.join(self.seg_dir, basename + '_train_id.png')
            if os.path.exists(seg_path):
                seg = Image.open(seg_path)
                seg = seg.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.NEAREST)
                seg = torch.from_numpy(np.array(seg)).long()
            else:
                seg = torch.zeros(INPUT_SIZE, dtype=torch.long)
            sample['seg'] = seg

        if 'depth' in self.tasks:
            depth_path = self._find_depth(basename)
            if depth_path:
                depth = Image.open(depth_path).convert('L')
                depth = depth.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
                depth = torch.from_numpy(np.array(depth, dtype=np.float32)).unsqueeze(0)
                depth = depth / 255.0
            else:
                depth = torch.zeros((1, *INPUT_SIZE), dtype=torch.float32)
            sample['depth'] = depth

        if 'lanes' in self.tasks:
            lane_path = os.path.join(self.lane_dir, basename + '.png')
            if os.path.exists(lane_path):
                lane = Image.open(lane_path)
                lane = lane.resize(
                    (INPUT_SIZE[1] // 4, INPUT_SIZE[0] // 4), Image.NEAREST)
                lane = torch.from_numpy(np.array(lane)).long()
            else:
                lane = torch.zeros(
                    (INPUT_SIZE[0] // 4, INPUT_SIZE[1] // 4), dtype=torch.long)
            sample['lanes'] = lane

        return sample


def get_dataloaders(root_dir, tasks=None, batch_size=4, num_workers=2,
                    seg_root=None, depth_root=None):
    """Create train and val dataloaders for BDD100K."""
    train_dataset = BDD100KDataset(
        root_dir, split='train', tasks=tasks,
        seg_root=seg_root, depth_root=depth_root,
    )
    val_dataset = BDD100KDataset(
        root_dir, split='val', tasks=tasks,
        seg_root=seg_root, depth_root=depth_root,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
