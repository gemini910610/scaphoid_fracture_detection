import torch

from datasets import ScaphoidDataset
from models import FasterRCNN
from rich_tqdm import tqdm, trange
from torch import optim
from torch.utils.data import DataLoader

dataset = ScaphoidDataset('AP')
loader = DataLoader(dataset, 1, True)

model = FasterRCNN().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)

epochs = 5000
total_loss = []
for epoch in trange(epochs):
    losses = 0
    for images, bboxes, labels, filenames in loader:
        images = images.cuda()
        bboxes = bboxes.cuda()
        labels = labels.cuda()

        images = list(images)
        targets = [
            {
                'boxes': bbox,
                'labels': label
            }
            for bbox, label in zip(bboxes, labels)
        ]

        loss_dict = model(images, targets)
        classifier_loss = loss_dict['loss_classifier']
        box_regression_loss = loss_dict['loss_box_reg']
        objectness_loss = loss_dict['loss_objectness']
        rpn_box_regression_loss = loss_dict['loss_rpn_box_reg']
        loss = classifier_loss + box_regression_loss + objectness_loss + rpn_box_regression_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
    
    average_loss = losses / dataset.length
    tqdm.write(f'[{epoch+1:>4}/{epochs}] Loss: {average_loss:.6f}')
    total_loss.append(average_loss)

state_dict = model.state_dict()
torch.save({
    'model': state_dict,
    'loss': total_loss
}, f'model_{epochs}.pth')
