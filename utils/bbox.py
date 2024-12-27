import numpy

def shift_bbox(bbox, dx, dy):
    bbox = numpy.array(bbox)
    bbox = bbox + [dx, dy]
    bbox = bbox.tolist()
    return bbox
