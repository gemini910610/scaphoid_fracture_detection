import json
import os
import re

def create_folders(dir, folders):
    if isinstance(folders, str):
        folders = [folders]
    for folder in folders:
        os.makedirs(f'{dir}/{folder}', exist_ok=True)

class ReNamer:
    def __init__(self):
        self.pattern = r'^(\d+).?([LR])?.*[ -](\S+)\..{3,4}'
        self.new_sides = {
            'AP': 'AP',
            'AP0': 'AP',
            'AP20': 'AP',
            'LA0': 'LA',
            'LAT0': 'LA',
            'LAT1': 'LA',
            'LAT20': 'LA',
            'OB0': 'AP',
            'OB20': 'AP',
            'RAP0': 'R_AP',
            'RLA0': 'R_LA',
            'SC0': 'AP'
        }
    def split_filename(self, filename):
        number, hand, side = re.search(self.pattern, filename).groups()
        return number, hand, side
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
        new_side = self.new_sides[side]
        new_filename = self.get_new_filename(number, hand, new_side)
        return new_filename
    def convert_extension(self, filename, extension):
        index = filename.find('.')
        filename = f'{filename[:index]}.{extension}'
        return filename

class JsonOpener:
    @staticmethod
    def read(filename):
        with open(filename) as file:
            annotation = json.load(file)
        return annotation
    @staticmethod
    def write(filename, annotation, indent=4):
        with open(filename, 'w') as file:
            json.dump(annotation, file, indent=indent)
