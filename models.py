from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=2, anchor_sizes=(96, 112, 128), anchor_aspect_ratios=(1 / 1.5, 1, 1.5)):
        # num_classes include background
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

        anchor_count = len(anchor_sizes) * len(anchor_aspect_ratios)

        self.model.anchor_generator = AnchorGenerator(
            sizes=[anchor_sizes],
            aspect_ratios=[anchor_aspect_ratios]
        )

        self.model.rpn.head = RPNHead(256, anchor_count)
    def forward(self, *args, **kwargs):
        forward_function = self.train_forward if self.training else self.eval_forward
        return_value = forward_function(*args, **kwargs)
        return return_value
    def train_forward(self, images, bboxes, labels):
        targets = self.get_model_input(bboxes, labels)
        losses = self.model(images, targets)
        return losses
    def eval_forward(self, images):
        predicts = self.model(images)
        return predicts
    def get_model_input(self, bboxes, labels):
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
    from torch.utils.data import DataLoader
    from utils.table import Table

    dataset = ScaphoidDataset()
    loader = DataLoader(dataset, 1, True)

    model = FasterRCNN().cuda()

    for images, bboxes, labels, filenames in loader:
        images = images.cuda()
        bboxes = bboxes.cuda()
        labels = labels.cuda()
        break

    losses = model(images, bboxes, labels)

    table = Table(
        title='FasterRCNN(train)',
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
        title='FasterRCNN(eval)',
        headers=['Object', 'Shape'],
        contents={
            'Bounding Box': tuple(boxes.shape),
            'Label': tuple(labels.shape),
            'Score': tuple(scores.shape)
        }
    )
    table.display()
