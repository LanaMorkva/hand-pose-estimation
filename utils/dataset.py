import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import projectPoints, create_heatmaps


RESIZED_IMG_SIZE = 128

class FreiHANDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = sorted(os.listdir(os.path.join(data_dir, 'rgb')))

        # Load annotations
        with open('FreiHAND_dataset/training_xyz.json', 'r') as f:
            self.xyz = np.array(json.load(f))

        with open('FreiHAND_dataset/training_K.json', 'r') as f:
            self.K = np.array(json.load(f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'rgb', self.image_paths[idx])
        image_org = Image.open(img_path).convert('RGB')
        img_org_size = image_org.size

        to_tensor = transforms.ToTensor()

        if self.transform:
            image = self.transform(image_org)
        else:
            image = to_tensor(image_org)

        anno_idx = idx % len(self.xyz)
        keypoints_3d = self.xyz[anno_idx]  
        camera_intrinsics = self.K[anno_idx] 

        keypoints = projectPoints(keypoints_3d, camera_intrinsics)
        keypoints = keypoints / img_org_size
        heatmaps = create_heatmaps(keypoints, RESIZED_IMG_SIZE)
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        target = {
            'keypoints': keypoints,
            'heatmaps': heatmaps,
            'img_size': img_org_size,
            'original': to_tensor(image_org)
        }

        return image, target
    

    def visualize(self, idx):
        _, target = self[idx]

        img_size = target['img_size']
        keypoints = np.array(target['keypoints']) * np.array(img_size)

        image = target['original']

        _, ax = plt.subplots()
        ax.imshow(image)

        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=10, marker='.', c='r')
        for i, point in enumerate(keypoints):
            ax.annotate(str(i), (point[0], point[1]))

        plt.show()

        self.plot_heatmaps(target['heatmaps'], 'heatmaps')

    
    def plot_heatmaps(self, heatmaps, filename):
        _, axes = plt.subplots(3, 7, figsize=(15, 10))
        axes = list(axes)
        for i, heatmap in enumerate(heatmaps):
            axes[i // 7][i % 7].imshow(heatmap, cmap='jet')
            axes[i // 7][i % 7].axis('off')
        plt.tight_layout()
        plt.savefig('figures/' + filename)
