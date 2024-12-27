import torch

from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=2, anchor_sizes=None, anchor_aspect_ratios=None, device='cuda'):
        # num_classes include background
        super().__init__()
        if anchor_sizes is None:
            anchor_sizes = [[32], [64], [128], [256], [512]]
        if anchor_aspect_ratios is None:
            anchor_aspect_ratios = [[0.5, 1, 2]] * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios)
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes, rpn_anchor_generator=anchor_generator)
        self.to(device)
    def forward(self, *args, **kwargs):
        forward_function = self.train_forward if self.training else self.eval_forward
        return_value = forward_function(*args, **kwargs)
        return return_value
    def train_forward(self, images, bboxes, labels):
        targets = self.format_targets(bboxes, labels)
        losses = self.model(images, targets)
        return losses
    def eval_forward(self, images, *, gradient=False):
        grad_mode = torch.enable_grad if gradient else torch.no_grad
        with grad_mode():
            predicts = self.model(images)
        return predicts
    def format_targets(self, bboxes, labels):
        targets = [
            {
                'boxes': bbox,
                'labels': label
            }
            for bbox, label in zip(bboxes, labels)
        ]
        return targets

if __name__ == '__main__':
    from datasets import ScaphoidDataset
    from utils.table import Table

    loader = ScaphoidDataset.create_loader()

    model = FasterRCNN()

    for images, bboxes, labels, filenames in loader:
        break

    losses = model(images, bboxes, labels)

    table = Table(
        title=f'{model.__class__.__name__}(train)',
        headers=['Loss', 'Value'],
        contents={
            'RPN Classifier': f'{losses["loss_objectness"].item():.6f}',
            'RPN Box Regressor': f'{losses["loss_rpn_box_reg"].item():.6f}',
            'Classifier': f'{losses["loss_classifier"].item():.6f}',
            'Box Regressor': f'{losses["loss_box_reg"].item():.6f}'
        }
    )
    table.display()

    model.eval()

    predicts = model(images)
    predict = predicts[0]
    boxes = predict['boxes']
    labels = predict['labels']
    scores = predict['scores']

    table = Table(
        title=f'{model.__class__.__name__}(eval)',
        headers=['Object', 'Shape'],
        contents={
            'Bounding Box': tuple(boxes.shape),
            'Label': tuple(labels.shape),
            'Score': tuple(scores.shape)
        }
    )
    table.display()
