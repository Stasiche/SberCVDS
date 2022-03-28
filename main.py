from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
import torch
import wandb
from src.utils import train_one_epoch, eval_model, save_model
import torchvision.models as models
# convnext_tiny, efficientnet_b0, efficientnet_b5


torch.manual_seed(0)
wandb.init(project='SberCVDS',
           config={
               'batch_size': 3,
               'lr': 1e-4,
               'model_name': 'efficientnet_b5',
               'epochs': 20,
               'grad_accum': 15
           })


config = wandb.config
device = torch.device('cuda')
train_dataset = ImageWoofDataset('train')
val_dataset = ImageWoofDataset('val')
traindata = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(val_dataset, batch_size=50, shuffle=False)

model = getattr(models, config.model_name)(pretrained=True)
model.classifier[-1] = Linear(model.classifier[-1].in_features, 10)
model.to(device)
optimizer = Adam(model.parameters(), lr=config.lr)
criterion = CrossEntropyLoss()

step = 0
eval_model(model, valdata, step)
for epoch in range(config.epochs):
    step = train_one_epoch(model, optimizer, criterion, epoch, step, traindata, config.grad_accum)
    eval_model(model, valdata, step)
    save_model(model)
wandb.finish()
