import torch

from shapely import Polygon
from torch import nn

class IoU(nn.Module):
    def forward(self, predicts, targets):
        metrics = []
        for predict, target in zip(predicts, targets):
            if len(predict['boxes']) == 0:
                metrics.append(0)
                continue

            predict = predict['boxes'][0]
            target = target[0]

            predict_polygon = self.construct_polygon(predict)
            target_polygon = self.construct_polygon(target)

            intersection = predict_polygon.intersection(target_polygon).area
            union = predict_polygon.area + target_polygon.area - intersection
            iou = intersection / union

            metrics.append(iou)
        metrics = torch.tensor(metrics)
        return metrics
    def construct_polygon(self, bbox):
        left, top, right, bottom = bbox
        polygon = Polygon([
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ])
        return polygon
