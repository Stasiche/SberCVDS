from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import wandb
from src.utils import train_one_epoch, eval_model, save_model

torch.manual_seed(0)
wandb.init(project='SberCVDS',
           config={
               'batch_size': 15,
               'lr': 1e-4,
               'eff_model_name': 'efficientnet-b0',
               'epochs': 10,
           })
config = wandb.config
device = torch.device('cuda')
train_dataset = ImageWoofDataset('train')
val_dataset = ImageWoofDataset('val')
traindata = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(val_dataset, batch_size=50, shuffle=False)

model = EfficientNet.from_pretrained(config.eff_model_name, num_classes=10).to(device)
optimizer = Adam(model.parameters(), lr=config.lr)
criterion = CrossEntropyLoss()

step = 0
eval_model(model, valdata, step)
for epoch in range(config.epochs):
    step = train_one_epoch(model, optimizer, criterion, epoch, step, traindata)
    eval_model(model, valdata, step)
    save_model(model)
wandb.finish()
