import albumentations
import cv2
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from utils.annotation import Annotation

class IdentityAugmentation:
    def __call__(self, **kwargs):
        return kwargs

class ScaphoidDataset(Dataset):
    def __init__(self, mode='train', image_size = None, augmentation=True):
        self.mode = mode
        self.filenames = os.listdir(f'./dataset/{mode}')
        self.image_size = image_size if image_size is not None else (1400, 1200)
        self.length = len(self.filenames)

        if augmentation:
            self.augmentation = albumentations.Compose([
                albumentations.CLAHE(always_apply=True),
                albumentations.HorizontalFlip(),
                albumentations.RandomBrightnessContrast(brightness_limit=0)
            ], albumentations.BboxParams('pascal_voc', []))
        else:
            self.augmentation = IdentityAugmentation()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size)
        ])
    def __getitem__(self, index):
        filename = self.filenames[index]
        annotation = Annotation.load(f'./dataset/{self.mode}/{filename}')

        image_filename = annotation.image
        scaphoid_bbox = annotation.scaphoid_bbox

        image = cv2.imread(f'./ip_homework_data/scaphoid_detection/images/{image_filename}')
        origin_height, origin_width, _ = image.shape

        augmentations = self.augmentation(image=image, bboxes=[scaphoid_bbox])

        image = augmentations['image']
        image = self.transform(image)
        resized_height, resized_width = self.image_size

        bbox = augmentations['bboxes']
        bbox = torch.tensor(bbox)

        scale_height = resized_height / origin_height
        scale_width = resized_width / origin_width
        scale = torch.tensor([scale_width, scale_height, scale_width, scale_height])

        bbox = bbox * scale

        label = torch.ones(1, dtype=torch.long)

        return image, bbox, label, filename
    def __len__(self):
        return self.length

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils.table import Table

    dataset = ScaphoidDataset()
    loader = DataLoader(dataset, 2, True)
    for images, bboxes, labels, filenames in loader:
        break

    table = Table(
        title='Scaphoid',
        headers=['Object', 'Content'],
        contents={
            'Image': tuple(images.shape),
            'Bounding Box': tuple(bboxes.shape),
            'Label': tuple(labels.shape),
            'Filename': f'({len(filenames)},)'
        }
    )
    table.display()
