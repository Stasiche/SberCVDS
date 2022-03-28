import wandb
from os.path import join, exists, dirname
import torch
import torchvision.models as models
from torch.nn import Linear

from time import perf_counter
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


@torch.no_grad()
def eval_model(model, valdata):
    device = next(model.parameters()).device
    model.eval()
    acc = 0
    predicts = []
    gts = []
    for batch_num, (batch, labels) in enumerate(tqdm(valdata)):
        batch = batch.to(device)
        labels = labels.to(device)

        predict = torch.argmax(model(batch), dim=1)
        acc += sum(labels == predict).item()

        predicts.extend(predict.tolist())
        gts.extend(labels.tolist())

    return acc / len(valdata.dataset), np.vstack([predicts, gts])


def calc_parameters(model):
    return sum(p.numel() for p in model.parameters())


def plot_errors(predicted_label, gt_label, img_path, save_path, convector):
    fig, ax = plt.subplots(1, figsize=(6, 6,))
    ax.imshow(Image.open(img_path))
    ax.axis('off')
    ax.title.set_text(
        f'Predicted: {convector.index_to_breed[predicted_label]}\nGround truth: {convector.index_to_breed[gt_label]}')
    fig.savefig(f'{save_path}.jpeg', bbox_inches='tight')


def load_model(run):
    model = getattr(models, run.config['model_name'])()
    model.classifier[-1] = Linear(model.classifier[-1].in_features, 10)
    restore_model(model, run.id)
    model.eval()
    return model


@contextmanager
def stopwatch() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def restore_model(model, run_name):
    local_model_path = join('models', run_name, 'model.pt')

    if not exists(local_model_path):
        wandb.restore('model.pt', f'stasiche/SberCVDS/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
