import os

from utils.annotation import ScaphoidAnnotation
from utils.bbox import shift_bbox
from utils.data import DataSplitter
from utils.file import create_folders, JsonOpener, ReNamer
from utils.rich_tqdm import tqdm

if __name__ == '__main__':
    dataset_dir = 'datasets/scaphoid_dataset'
    create_folders(dataset_dir, ['train', 'val'])

    origin_data_dir = 'ip_homework_data'
    extra_data_dir = 'ip_homework_data'

    filenames = []

    re_namer = ReNamer()

    data_dirs = ['ip_homework_data', 'ip_homework_add']
    for data_dir in data_dirs:
        origin_filenames = os.listdir(f'{data_dir}/scaphoid_detection/images')
        for origin_filename in origin_filenames:
            new_filename = re_namer.rename(origin_filename)
            if new_filename.endswith('LA.jpg'):
                continue
            filenames.append({
                'origin': origin_filename,
                'new': new_filename,
                'dir': data_dir
            })

    data_splitter = DataSplitter()
    train_filenames, val_filenames = data_splitter.train_val_split(filenames)

    for mode, filenames in {'train': train_filenames, 'val': val_filenames}.items():
        for filename in tqdm(filenames, desc=f'{mode:<5}'):
            image_filename = filename['origin']
            new_image_filename = filename['new']
            data_dir = filename['dir']

            annotation = ScaphoidAnnotation(f'{data_dir}/scaphoid_detection/images/{image_filename}')

            # annotations
            annotation_filename = re_namer.convert_extension(image_filename, 'json')
            new_annotation_filename = re_namer.convert_extension(new_image_filename, 'json')

            ## scaphoid annotations
            scaphoid_annotation = JsonOpener.read(f'{data_dir}/scaphoid_detection/annotations/{annotation_filename}')
            scaphoid_bbox = scaphoid_annotation[0]['bbox']
            scaphoid_bbox = [int(value) for value in scaphoid_bbox]
            annotation.set_scaphoid_bbox(scaphoid_bbox)

            ## fracture annotations
            fracture_annotation = JsonOpener.read(f'{data_dir}/fracture_detection/annotations/{annotation_filename}')
            fracture_bbox = fracture_annotation[0]['bbox']
            if fracture_bbox is not None:
                left = scaphoid_bbox[0]
                top = scaphoid_bbox[1]
                fracture_bbox = shift_bbox(fracture_bbox, left, top)
                annotation.set_fracture_bbox(fracture_bbox)

            annotation.save(f'{dataset_dir}/{mode}/{new_annotation_filename}')
