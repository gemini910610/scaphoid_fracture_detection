import os
import torch

from datasets import ScaphoidDataset
from metrics import IoU
from models import FasterRCNN
from rich_tqdm import tqdm, trange
from torch import optim
from torch.utils.data import DataLoader, random_split

def run_epoch(model, loader, metric, train, *, optimizer=None):
    losses = 0
    metrics = 0
    for images, bboxes, labels, _ in loader:
        images = images.cuda()
        bboxes = bboxes.cuda()
        labels = labels.cuda()

        targets = [
            {
                'boxes': bbox,
                'labels': label
            }
            for bbox, label in zip(bboxes, labels)
        ]

        if train:
            loss_dict = model(images, targets)
        else:
            with torch.no_grad():
                loss_dict = model(images, targets)

        loss = sum(loss_dict.values())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses += loss.item()

        model.eval()
        with torch.no_grad():
            predicts = model(images)
        model.train()

        iou = metric(predicts, bboxes)
        metrics += iou.sum().item()
    
    average_loss = losses / len(loader.dataset)
    average_metric = metrics / len(loader.dataset)

    return average_loss, average_metric

os.makedirs('./experiments', exist_ok=True)

dataset = ScaphoidDataset('AP')
train_dataset, eval_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, 1, True)
eval_loader = DataLoader(eval_dataset, 1, True)

model = FasterRCNN().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)

metric = IoU().cuda()

epochs = 5000
total_loss = {
    'train': [],
    'eval': []
}
total_metric = {
    'train': [],
    'eval': []
}
for epoch in trange(epochs):
    train_loss, train_metric = run_epoch(model, train_loader, metric, True, optimizer=optimizer)
    total_loss['train'].append(train_loss)
    total_metric['train'].append(train_metric)
    tqdm.write(f'[{epoch+1:>4}/{epochs}]')
    tqdm.write(f'Train Loss: {train_loss:.6f}, IoU: {train_metric:.6f}')

    eval_loss, eval_metric = run_epoch(model, eval_loader, metric, False)
    total_loss['eval'].append(eval_loss)
    total_metric['eval'].append(eval_metric)
    tqdm.write(f' Eval Loss: {eval_loss:.6f}, IoU: {eval_metric:.6f}')

    if (epoch + 1) % 1000 == 0:
        state_dict = model.state_dict()
        torch.save({
            'model': state_dict,
            'loss': total_loss
        }, f'./experiments/model_{epoch+1}.pth')
