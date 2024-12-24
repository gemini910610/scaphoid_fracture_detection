from utils.files import JsonOpener

class Annotation:
    def __init__(self, filename):
        self.image = filename
        self.scaphoid_bbox = None
        self.fracture_bbox = None
    def set_scaphoid_bbox(self, bbox):
        self.scaphoid_bbox = bbox
    def set_fracture_bbox(self, bbox):
        self.fracture_bbox = bbox
    def format_data(self):
        data = {
            'image': self.image,
            'bboxes': {
                'scaphoid': self.scaphoid_bbox,
                'fracture': self.fracture_bbox
            }
        }
        return data
    def save(self, filename):
        data = self.format_data()
        JsonOpener.write(filename, data)
    @staticmethod
    def load(filename):
        data = JsonOpener.read(filename)
        image = data['image']
        scaphoid_bbox = data['bboxes']['scaphoid']
        fracture_bbox = data['bboxes']['fracture']

        annotation = Annotation(image)
        annotation.set_scaphoid_bbox(scaphoid_bbox)
        annotation.set_fracture_bbox(fracture_bbox)
        return annotation
    def __str__(self):
        data = self.format_data()
        return str(data)
