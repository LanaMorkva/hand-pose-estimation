import numpy as np 
import cv2
import matplotlib.pyplot as plt
import math

import torch

def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def gaussian_heatmap(size, center, sigma):
    y, x = np.arange(size), np.arange(size)
    x, y = np.meshgrid(x, y)
    center_x, center_y = center
    return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

def create_heatmaps(keypoints, size, sigma=3):
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, size, size), dtype=np.float32)
    
    for i, (x, y) in enumerate(keypoints): 
        x_scaled, y_scaled = int(x * size), int(y * size)
        if 0 <= x < size and 0 <= y < size:
            heatmaps[i] = gaussian_heatmap(size, (x_scaled, y_scaled), sigma)
    
    return torch.tensor(heatmaps) 

def plot_losses(train_loss, val_loss, filename):
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.savefig('figures/' + filename)

def plot_heatmaps(heatmaps, filename):
    fig, axes = plt.subplots(math.ceil(len(heatmaps) / 6), 6, figsize=(10, 10))
    axes = list(axes)
    for i, heatmap in enumerate(heatmaps):
        axes[i // 6][i % 6].imshow(heatmap, cmap='jet')
        axes[i // 6][i % 6].axis('off')
    plt.tight_layout()
    plt.savefig('figures/' + filename)

def plot_results(targets, heatmaps, resized_img_size, filename):
    pred_keypoints = torch.argmax(heatmaps, dim=2)
    # find exact location
    pred_keypoints_y = pred_keypoints // resized_img_size
    pred_keypoints_x = pred_keypoints - resized_img_size * pred_keypoints_y

    # normalize positions
    pred_keypoints_x /= resized_img_size
    pred_keypoints_y /= resized_img_size

    num_images = len(targets)
    _, axes = plt.subplots(math.ceil(num_images / 5), 5, figsize=(15, 10))
    axes = list(axes)
    for i, target in enumerate(targets):
        img_size = target['img_size']
        img_org = target['original']

        image = np.array(img_org)
        image = image.transpose((1, 2, 0))

        keypoints_x = pred_keypoints_x[i] * img_size
        keypoints_y = pred_keypoints_y[i] * img_size

        axes[i // 5][i % 5].imshow(image)
        axes[i // 5][i % 5].scatter(keypoints_x, keypoints_y, s=10, marker='.', c='r')
        axes[i // 5][i % 5].axis('off')
    plt.tight_layout()
    plt.savefig('figures/' + filename)

def find_heatmap_peak(heatmap, resized_img_size):
    max_idx = torch.argmax(heatmap)

    center_y, center_x = divmod(max_idx.item(), resized_img_size)    
    return center_x, center_y

def keypoints_from_heatmaps(heatmaps, resized_img_size):
    pred_keypoints = []
    for heatmap in heatmaps:
        pred_keypoints.append(find_heatmap_peak(heatmap, resized_img_size))

    pred_keypoints = torch.tensor(pred_keypoints).float()
    pred_keypoints /= resized_img_size

    return pred_keypoints

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.EPSILON = 1e-6

    def forward(self, y_pred, y_true):
        pred_flat = y_pred.view(-1)
        target_flat = y_true.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = torch.sum(target_flat * target_flat) + torch.sum(pred_flat * pred_flat)
        dice_score = (2. * intersection + self.EPSILON) / (union + self.EPSILON)

        return 1 - torch.mean(dice_score)
    
