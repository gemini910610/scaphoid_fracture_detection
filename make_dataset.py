import json
import os
import re
import torch

from PIL import Image
from rich_tqdm import tqdm

data_dir = './ip_homework_data'

image_dir = f'{data_dir}/scaphoid_detection/images'
scaphoid_annotation_dir = f'{data_dir}/scaphoid_detection/annotations'
fracture_annotation_dir = f'{data_dir}/fracture_detection/annotations'

new_scaphoid_dir = './dataset/scaphoid_detection'
new_fracture_dir = './dataset/fracture_detection'

os.makedirs(f'{new_scaphoid_dir}/images', exist_ok=True)
os.makedirs(f'{new_scaphoid_dir}/annotations', exist_ok=True)
os.makedirs(f'{new_fracture_dir}/annotations', exist_ok=True)

new_sides = {
    'AP0': 'AP',
    'LA0': 'LA',
    'LAT0': 'LA',
    'LAT20': 'LA',
    'OB0': 'AP',
    'RAP0': 'R_AP',
    'RLA0': 'R_LA',
    'SC0': 'AP'
}

files = os.listdir(image_dir)
for image_filename in tqdm(files):
    # rename images
    pattern = r'^(\d+).?([LR])?.*[ -](\S+)\.jpg'
    id, hand, side = re.search(pattern, image_filename).groups()
    new_side = new_sides[side]
    new_image_filename = f'{id}_{new_side}.jpg' if hand is None else f'{id}_{hand}_{new_side}.jpg'
    image = Image.open(f'{image_dir}/{image_filename}')
    image.save(f'{new_scaphoid_dir}/images/{new_image_filename}')
    width = image.width
    height = image.height
    image.close()

    # rename annotations
    filename = image_filename.replace('jpg', 'json')
    filename = filename.replace('OB', 'AP')
    new_filename = new_image_filename.replace('jpg', 'pth')
    # scaphoid annotations
    with open(f'{scaphoid_annotation_dir}/{filename}') as file:
        scaphoid_annotation = json.load(file)[0]
    scaphoid_bbox = scaphoid_annotation['bbox']
    scaphoid_bbox = torch.tensor([int(box) for box in scaphoid_bbox])
    new_scaphoid_bbox = scaphoid_bbox / torch.tensor([width, height, width, height])
    with open(f'{new_scaphoid_dir}/annotations/{new_filename}', 'wb') as file:
        torch.save(new_scaphoid_bbox, file)
    # fracture annotations
    with open(f'{fracture_annotation_dir}/{filename}') as file:
        fracture_annotation = json.load(file)[0]
    if fracture_annotation['name'] == 'Fracture':
        fracture_bbox = torch.tensor(fracture_annotation['bbox'])
        new_fracture_bbox = fracture_bbox + scaphoid_bbox[:2]
    else:
        new_fracture_bbox = torch.tensor([])
    with open(f'{new_fracture_dir}/annotations/{new_filename}', 'wb') as file:
        torch.save(new_fracture_bbox, file)
