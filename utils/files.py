import json
import os

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

def create_folders(dir, folders):
    for folder in folders:
        os.makedirs(f'{dir}/{folder}', exist_ok=True)
