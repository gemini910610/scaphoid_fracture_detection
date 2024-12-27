import albumentations
import cv2
import os
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.annotation import ScaphoidAnnotation

class ScaphoidDataset(Dataset):
    def __init__(self, mode='train', image_size=(1400, 1200), augmentation=True):
        self.mode = mode
        self.filenames = os.listdir(f'datasets/scaphoid_dataset/{mode}')
        self.image_size = image_size
        self.length = len(self.filenames)

        if augmentation:
            self.augmentation = albumentations.Compose([
                albumentations.CLAHE(always_apply=True),
                albumentations.HorizontalFlip(),
                albumentations.RandomBrightnessContrast(brightness_limit=0)
            ], albumentations.BboxParams('pascal_voc', []))
        else:
            self.augmentation = albumentations.CLAHE(always_apply=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size)
        ])
    def __getitem__(self, index):
        filename = self.filenames[index]
        annotation = ScaphoidAnnotation.load(f'datasets/scaphoid_dataset/{self.mode}/{filename}')

        image_filename = annotation.image
        scaphoid_bbox = annotation.scaphoid_bbox

        image = cv2.imread(image_filename)
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
    @staticmethod
    def create_loader(mode='train', image_size=(1400, 1200), augmentation=True, batch_size=1, shuffle=True):
        dataset = ScaphoidDataset(mode, image_size, augmentation)
        loader = DataLoader(dataset, batch_size, shuffle)
        return loader

if __name__ == '__main__':
    from utils.table import Table

    loader = ScaphoidDataset.create_loader(batch_size=8)
    for images, bboxes, labels, filenames in loader:
        break

    table = Table(
        title='Scaphoid Dataset',
        headers=['Object', 'Shape'],
        contents={
            'Image': tuple(images.shape),
            'Bounding Box': tuple(bboxes.shape),
            'Label': tuple(labels.shape),
            'Filename': f'({len(filenames)},)'
        }
    )
    table.display()
