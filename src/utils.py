import torch
import wandb
from os.path import join


def train_one_epoch(model, optimizer, criterion, epoch, step, traindata, grad_accum):
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

            wandb.log({'loss': total_loss}, step=step)
            total_loss = 0
        wandb.log({'epoch': epoch + batch_num / len(traindata)}, step=step)
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
