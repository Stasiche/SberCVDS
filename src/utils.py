import torch
import wandb
from os.path import join, exists, dirname


def train_one_epoch(model, optimizer, criterion, epoch, step, traindata):
    device = next(model.parameters()).device
    model.train()
    for batch_num, (batch, labels) in enumerate(traindata):
        step += 1
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        wandb.log({'loss': loss.item(), 'epoch': epoch + batch_num / len(traindata)}, step=step)
    return step


@torch.no_grad()
def eval_model(model, valdata, step):
    device = next(model.parameters()).device
    model.eval()
    acc = 0
    for batch_num, (batch, labels) in enumerate(valdata):
        batch = batch.to(device)
        labels = labels.to(device)

        predict = torch.argmax(model(batch), dim=1)
        acc += sum(labels == predict).item()
    wandb.log({'accuracy': acc / len(valdata.dataset)}, step=step)


def save_model(model):
    torch.save(model.state_dict(), join(wandb.run.dir, 'model.pt'))
    wandb.save(join(wandb.run.dir, 'model.pt'))


def restore_model(model, run_name):
    local_model_path = join('..', 'models', 'model.pt')

    if not exists(local_model_path):
        wandb.restore('model.pt', f'stasiche/SberCVDS/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(local_model_path))
