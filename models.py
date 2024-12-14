from torchvision.models.detection import fasterrcnn_resnet50_fpn

def FasterRCNN(num_classes=2):
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    return model

if __name__ == '__main__':
    from datasets import ScaphoidDataset
    from table import Table
    from torch.utils.data import DataLoader

    dataset = ScaphoidDataset('AP')
    loader = DataLoader(dataset, 1, True)

    for images, bboxes, labels, filenames in loader:
        images = images.cuda()
        bboxes = bboxes.cuda()
        labels = labels.cuda()
        break

    images = list(images)
    targets = [
        {
            'boxes': bbox,
            'labels': label
        }
        for bbox, label in zip(bboxes, labels)
    ]

    model = FasterRCNN().cuda()

    losses = model(images, targets)
    
    table = Table(
        title='FasterRCNN(train)',
        headers=['Loss', 'Value'],
        contents={
            'Classifier': f"{losses['loss_classifier'].item():.6f}",
            'Box Regressor': f"{losses['loss_box_reg'].item():.6f}",
            'Objectness': f"{losses['loss_objectness'].item():.6f}",
            'RPN Box Regressor': f"{losses['loss_rpn_box_reg'].item():.6f}"
        }
    )
    table.display()
