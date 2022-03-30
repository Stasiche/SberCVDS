import torch
import wandb
from os.path import join, dirname
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.nn import CrossEntropyLoss


def train_one_epoch(model, optimizer, criterion, epoch, step, traindata, grad_accum, scheduler):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0
    for batch_num, (batch, labels) in enumerate(traindata):
        step += 1
        batch = batch.to(device)
        labels = labels.to(device)

        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss /= grad_accum
        loss.backward()

        total_loss += loss.item()
        if not step % grad_accum:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                wandb.log({'lr': scheduler.optimizer.param_groups[0]['lr']}, step=step)

            wandb.log({'loss': total_loss}, step=step)
            total_loss = 0
        wandb.log({'epoch': epoch + batch_num / len(traindata)}, step=step)
    return step


def calc_metrics(true: np.ndarray, predict: np.ndarray):
    acc = (true == predict).mean()
    f1_micro = f1_score(true, predict, average='micro')
    f1_macro = f1_score(true, predict, average='macro')
    mc = matthews_corrcoef(true, predict)

    return {'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'matthews': mc}


@torch.no_grad()
def eval_model(model, valdata, step):
    device = next(model.parameters()).device
    model.eval()
    gts = []
    predicts = []
    loss = 0
    criterion = CrossEntropyLoss()
    for batch_num, (batch, labels) in enumerate(valdata):
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = model(batch)

        loss += criterion(outputs, labels).item()
        predict = torch.argmax(outputs, dim=1)

        gts.extend(labels.tolist())
        predicts.extend(predict.tolist())
    predicts = np.array(predicts)
    gts = np.array(gts)
    metrics = calc_metrics(gts, predicts)

    wandb.log(metrics, step=step)
    wandb.log({'val_crossentropy': loss / len(valdata.dataset)}, step=step)


def save_model(model, name):
    torch.save(model.state_dict(), join(wandb.run.dir, f'{name}.pt'))
    wandb.save(join(wandb.run.dir, f'{name}.pt'))


def restore_model(model, project_name, run_name, model_name='model'):
    local_model_path = join('..', 'models', project_name, run_name, f'{model_name}.pt')

    wandb.restore(f'{model_name}.pt', f'stasiche/{project_name}/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(local_model_path,  map_location=torch.device('cpu')))
