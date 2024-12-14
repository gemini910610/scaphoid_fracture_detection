import json
import numpy
import os
import re
import shutil

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
    shutil.copyfile(f'{image_dir}/{image_filename}', f'{new_scaphoid_dir}/images/{new_image_filename}')

    # rename annotations
    filename = image_filename.replace('jpg', 'json')
    filename = filename.replace('OB', 'AP')
    new_filename = new_image_filename.replace('jpg', 'npy')
    # scaphoid annotations
    with open(f'{scaphoid_annotation_dir}/{filename}') as file:
        scaphoid_annotation = json.load(file)[0]
    scaphoid_bbox = scaphoid_annotation['bbox']
    scaphoid_bbox = numpy.array([int(box) for box in scaphoid_bbox])
    with open(f'{new_scaphoid_dir}/annotations/{new_filename}', 'wb') as file:
        numpy.save(file, scaphoid_bbox)
    # fracture annotations
    with open(f'{fracture_annotation_dir}/{filename}') as file:
        fracture_annotation = json.load(file)[0]
    if fracture_annotation['name'] == 'Fracture':
        fracture_bbox = numpy.array(fracture_annotation['bbox'])
        new_fracture_bbox = fracture_bbox + scaphoid_bbox[:2]
    else:
        new_fracture_bbox = numpy.array([])
    with open(f'{new_fracture_dir}/annotations/{new_filename}', 'wb') as file:
        numpy.save(file, new_fracture_bbox)
