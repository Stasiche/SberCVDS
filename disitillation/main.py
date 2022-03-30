from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss, Linear, KLDivLoss
from torch.optim import Adam
import wandb
import torchvision.models as models
from src.utils import restore_model, eval_model, save_model


def train_one_epoch(student, teacher, optimizer, criterion, epoch, step, traindata, grad_accum, alpha):
    device = next(student.parameters()).device
    student.train()
    total_loss, total_s_loss, total_distill_loss = 0, 0, 0
    distill_criterion = KLDivLoss()
    for batch_num, (batch, labels) in enumerate(traindata):
        step += 1
        batch = batch.to(device)
        labels = labels.to(device)

        outputs = student(batch)
        s_loss = criterion(outputs, labels)

        distill_loss = distill_criterion(outputs, teacher(batch))

        loss = s_loss + alpha*distill_loss
        loss /= grad_accum
        loss.backward()

        total_loss += loss.item()
        total_s_loss += s_loss.item()
        total_distill_loss += distill_loss.item()
        if not step % grad_accum:
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({'loss': total_loss, 's_loss': total_s_loss, 'distill_loss': total_distill_loss}, step=step)
            total_loss = 0
            total_s_loss = 0
            total_distill_loss = 0
        wandb.log({'epoch': epoch + batch_num / len(traindata)}, step=step)
    return step


torch.manual_seed(0)
wandb.init(project='SberCVDS',
           notes='distill',
           config={
               'batch_size': 12,
               'lr': 3e-4,
               'student_model_name': 'efficientnet_b1',
               'teacher_run_name': '1fx3trl2',
               'teacher_model_name': '5',
               'epochs': 30,
               'grad_accum': 60,
               'alpha': 1e-2
           })

config = wandb.config
device = torch.device('cuda')
train_dataset = ImageWoofDataset('train')
val_dataset = ImageWoofDataset('val')
traindata = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(val_dataset, batch_size=50, shuffle=False)

student = getattr(models, config.student_model_name)(pretrained=True)
student.classifier[-1] = Linear(student.classifier[-1].in_features, 10)
student.to(device)

api = wandb.Api()
run = api.run(f'stasiche/SberCVDS/{config.teacher_run_name}')
teacher = getattr(models, run.config['model_name'])(pretrained=True)
teacher.classifier[-1] = Linear(teacher.classifier[-1].in_features, 10)
restore_model(teacher, 'SberCVDS', config.teacher_run_name, config.teacher_model_name)
teacher.requires_grad = False
teacher.eval()
teacher.to(device)

optimizer = Adam(student.parameters(), lr=config.lr)
criterion = CrossEntropyLoss()

step = 0
# eval_model(model, valdata, step)
for epoch in range(config.epochs):
    step = train_one_epoch(student, teacher, optimizer, criterion, epoch, step, traindata, config.grad_accum,
                           config.alpha)
    eval_model(student, valdata, step)
    if not (epoch + 1) % 2:
        save_model(student, epoch)
wandb.finish()
