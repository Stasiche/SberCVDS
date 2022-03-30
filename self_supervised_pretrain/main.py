from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader
import torch
import wandb
# convnext_tiny, efficientnet_b0, efficientnet_b5
from btwins import BTWINS

torch.manual_seed(0)
wandb.init(project='SberCVDSss',
           config={
               'batch_size': 25,
               'lr': 3e-4,
               'model_name': 'efficientnet_b0',
               'epochs': 30,
               'grad_accum': 60,
               'scheduler': None
           })


config = wandb.config
device = torch.device('cpu')
train_dataset = ImageWoofDataset('train')
val_dataset = ImageWoofDataset('val')
traindata = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(val_dataset, batch_size=50, shuffle=False)

btwins = BTWINS(config.model_name)
btwins.fit(traindata, epochs=config.epochs, lr=config.lr)
wandb.finish()
