import torch
import wandb


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
