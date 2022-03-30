from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import wandb
from src.utils import train_one_epoch, eval_model, save_model, restore_model
import torchvision.models as models
# convnext_tiny, efficientnet_b0, efficientnet_b5


torch.manual_seed(0)
wandb.init(project='SberCVDS',
           config={
               'batch_size': 15,
               'lr': 3e-4,
               # 'mode': 'disabled',
               'model_name': 'efficientnet_b3',
               'epochs': 30,
               'grad_accum': 40,
               'scheduler': None,
               'ss_pretrain': None,
           })


config = wandb.config
device = torch.device('cuda')
train_dataset = ImageWoofDataset('train')
val_dataset = ImageWoofDataset('val')
traindata = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(val_dataset, batch_size=20, shuffle=False)

model = getattr(models, config.model_name)(pretrained=config.ss_pretrain is None)
if config.ss_pretrain is not None:
    restore_model(model, 'SberCVDSss', config.ss_pretrain, '6')
model.classifier[-1] = Linear(model.classifier[-1].in_features, 10)
model.to(device)
optimizer = Adam(model.parameters(), lr=config.lr)
criterion = CrossEntropyLoss()
if config.scheduler == 'cosine':
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000//config.grad_accum, T_mult=2)
else:
    scheduler = None

step = 0
# eval_model(model, valdata, step)
for epoch in range(config.epochs):
    step = train_one_epoch(model, optimizer, criterion, epoch, step, traindata, config.grad_accum, scheduler)
    eval_model(model, valdata, step)
    if not (epoch+1) % 2:
        save_model(model, epoch)
wandb.finish()
