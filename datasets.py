import albumentations
import cv2
import os
import torch

from table import Table
from torch.utils.data import Dataset
from torchvision import transforms

class ScaphoidDataset(Dataset):
    def __init__(self, side, image_size=None):
        self.filenames = [
            filename.replace('.jpg', '')
            for filename in os.listdir(f'./dataset/scaphoid_detection/images')
            if filename.endswith(f'{side}.jpg')
        ]
        self.length = len(self.filenames)
        self.augmentation = albumentations.Compose([
            albumentations.CLAHE(always_apply=True),
            albumentations.HorizontalFlip(),
            albumentations.RandomBrightnessContrast(brightness_limit=0)
        ], bbox_params=albumentations.BboxParams(format='albumentations', label_fields=[]))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size if image_size is not None else [1400, 1200])
        ])
    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(f'./dataset/scaphoid_detection/annotations/{filename}.pth', 'rb') as file:
            bbox = torch.load(file, weights_only=True)
        image = cv2.imread(f'./dataset/scaphoid_detection/images/{filename}.jpg')
        augmentations = self.augmentation(image=image, bboxes=[bbox])
        image = augmentations['image']
        bbox = torch.tensor(augmentations['bboxes'][0]).reshape(1, 4)
        image = self.transform(image)
        label = torch.ones(1, dtype=torch.int64)
        return image, bbox, label, filename
    def __len__(self):
        return self.length

if __name__ == '__main__':
    for side in ['AP', 'LA']:
        dataset = ScaphoidDataset(side)
        image, bbox, label, filename = dataset[0]
        table = Table(
            title=f'Scaphoid({side})',
            headers=['Object', 'Content'],
            contents={
                'Image': tuple(image.shape),
                'Bounding Box': tuple(bbox.shape),
                'Label': tuple(label.shape),
                'Filename': filename
            }
        )
        table.display()
