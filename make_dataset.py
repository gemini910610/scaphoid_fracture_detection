import json
import numpy
import os
import random
import re

from utils.files import create_folders, JsonOpener
from utils.rich_tqdm import tqdm
from utils.annotation import Annotation

class ReNamer:
    def __init__(self):
        self.pattern = r'^(\d+).?([LR])?.*[ -](\S+)\..{3,4}'
        self.new_side = {
            'AP0': 'AP',
            'LA0': 'LA',
            'LAT0': 'LA',
            'LAT20': 'LA',
            'OB0': 'AP',
            'RAP0': 'R_AP',
            'RLA0': 'R_LA',
            'SC0': 'AP'
        }
    def split_filename(self, filename):
        number, hand, side = re.search(self.pattern, filename).groups()
        return number, hand, side
    def get_new_side(self, side):
        new_side = self.new_side[side]
        return new_side
    def get_new_filename(self, number, hand, side):
        if hand is None:
            parts = [number, side]
        else:
            parts = [number, hand, side]
        filename = '_'.join(parts)
        filename = f'{filename}.jpg'
        return filename
    def rename(self, filename):
        number, hand, side = self.split_filename(filename)
        new_side = self.get_new_side(side)
        new_filename = self.get_new_filename(number, hand, new_side)
        return new_filename
    def convert_extension(self, filename, extension):
        index = filename.find('.')
        filename = f'{filename[:index]}.{extension}'
        return filename

class DataSplitter:
    def __init__(self, train_size=0.8, seed=42):
        self.train_size = train_size
        self.seed = seed
    def shuffle(self, data):
        random.seed(self.seed)
        data = random.sample(data, len(data))
        return data
    def train_val_split(self, data):
        data = self.shuffle(data)

        train_count = int(len(data) * self.train_size)
        train_data = data[:train_count]
        val_data = data[train_count:]
        return train_data, val_data

def shift_bbox(bbox, dx, dy):
    bbox = numpy.array(bbox)
    bbox = bbox + [dx, dy]
    bbox = bbox.tolist()
    return bbox

if __name__ == '__main__':
    data_dir = './ip_homework_data'
    image_dir = f'{data_dir}/scaphoid_detection/images'
    scaphoid_annotation_dir = f'{data_dir}/scaphoid_detection/annotations'
    fracture_annotation_dir = f'{data_dir}/fracture_detection/annotations'

    dataset_dir = './dataset'
    folders = ['train', 'val']
    create_folders(dataset_dir, folders)

    re_namer = ReNamer()
    image_filenames = os.listdir(image_dir)
    filenames = [
        (image_filename, new_image_filename)
        for image_filename in image_filenames
        if (new_image_filename := re_namer.rename(image_filename)).endswith('AP.jpg')
    ]

    data_splitter = DataSplitter()
    train_filenames, val_filenames = data_splitter.train_val_split(filenames)

    for mode, filenames in {'train': train_filenames, 'val': val_filenames}.items():
        for image_filename, new_image_filename in tqdm(filenames, desc=f'{mode:<5}'):
            annotation = Annotation(image_filename)

            # annotations
            annotation_filename = re_namer.convert_extension(image_filename, 'json')
            new_annotation_filename = re_namer.convert_extension(new_image_filename, 'json')

            ## scaphoid annotations
            scaphoid_annotation = JsonOpener.read(f'{scaphoid_annotation_dir}/{annotation_filename}')
            scaphoid_bbox = scaphoid_annotation[0]['bbox']
            scaphoid_bbox = [int(value) for value in scaphoid_bbox]
            annotation.set_scaphoid_bbox(scaphoid_bbox)

            ## fracture annotations
            fracture_annotation = JsonOpener.read(f'{fracture_annotation_dir}/{annotation_filename}')
            fracture_bbox = fracture_annotation[0]['bbox']
            if fracture_bbox is not None:
                left = scaphoid_bbox[0]
                top = scaphoid_bbox[1]
                fracture_bbox = shift_bbox(fracture_bbox, left, top)
                annotation.set_fracture_bbox(fracture_bbox)

            annotation.save(f'{dataset_dir}/{mode}/{new_annotation_filename}')
