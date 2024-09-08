import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from utils.utils_detection import get_annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class EgoHandDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None):
        self.img_dir = img_dir
        self.annotations = get_annotations(ann_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno_dict = self.annotations[idx]

        video_dir = anno_dict['video_name']
        img_name = 'frame_' + str(anno_dict['frame_num']).zfill(4) + '.jpg'

        # Load image
        img_path = os.path.join(self.img_dir, video_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        bboxes = anno_dict['boxes'] 
        bboxes = bboxes[~np.all(bboxes == 0, axis=1)]
        if self.transform:
            # image_size = np.array(image.size)
            # output_size = np.array(224, 224)
            # ratios = output_size / image_size
            # bboxes[:, 0::2] *= ratios[0]
            # bboxes[:, 1::2] *= ratios[1]
            image = self.transform(image)
        
        target = {
            'boxes':  torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor([1] * len(bboxes), dtype=torch.int64),
        }

        return image, target
    
    def visualize(self, idx):
        image, target = self[idx]
        image = np.array(image.cpu())
        image = image.transpose((1, 2, 0))
        boxes = np.array(target['boxes'])

        _, ax = plt.subplots()
        ax.imshow(image)
        for i, box in enumerate(boxes):  
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


        plt.show()