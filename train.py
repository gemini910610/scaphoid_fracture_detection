import torch

from datasets import ScaphoidDataset
from models import FasterRCNN
from torch import optim
from torch.utils.data import DataLoader
from utils.files import create_folders
from utils.metrics import IoU
from utils.rich_tqdm import tqdm, trange

def run_epoch(model, loader, metric, *, train=False, optimizer=None):
    losses = 0
    metrics = 0
    for images, bboxes, labels, _ in loader:
        images = images.cuda()
        bboxes = bboxes.cuda()
        labels = labels.cuda()

        if train:
            loss_dict = model(images, bboxes, labels)
        else:
            with torch.no_grad():
                loss_dict = model(images, bboxes, labels)
        
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
    
    average_loss = losses / loader.dataset.length
    average_metric = metrics / loader.dataset.length

    return average_loss, average_metric

create_folders('./experiments', ['scaphoid_detection'])

train_dataset = ScaphoidDataset('train')
train_loader = DataLoader(train_dataset, 1, True)
val_dataset = ScaphoidDataset('val')
val_loader = DataLoader(val_dataset, 1, True)

model = FasterRCNN().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

metric = IoU().cuda()

epochs = 600
save_frequency = 200

total_loss = {
    'train': [],
    'val': []
}
total_metric = {
    'train': [],
    'val': []
}

for epoch in trange(epochs):
    train_loss, train_metric = run_epoch(model, train_loader, metric, train=True, optimizer=optimizer)
    total_loss['train'].append(train_loss)
    total_metric['train'].append(train_metric)

    tqdm.write(f'[{epoch+1:>3}/{epochs}]')
    tqdm.write(f'Train Loss: {train_loss:.6f}, IoU: {train_metric:.6f}')

    val_loss, val_metric = run_epoch(model, val_loader, metric)
    total_loss['val'].append(val_loss)
    total_metric['val'].append(val_metric)

    tqdm.write(f'  Val Loss: {val_loss:.6f}, IoU: {val_metric:.6f}')

    if (epoch + 1) % save_frequency == 0:
        state_dict = model.state_dict()
        torch.save({
            'model': state_dict,
            'loss': total_loss,
            'metric': total_metric
        }, f'./experiments/scaphoid_detection/model_{epoch+1}.pth')
