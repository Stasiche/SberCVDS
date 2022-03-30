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
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


def plot_confusion_matrix(predicted_label, gt_label, save_path, convector):
    cf_matrix = confusion_matrix(gt_label, predicted_label)

    fig, ax = plt.subplots()
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    ax.set_xlabel('\nPredicted Breed')
    ax.set_ylabel('Actual Breed ')
    ax.tick_params(axis='x', rotation=80)
    ax.tick_params(axis='y', rotation=15)
    ax.tick_params(axis='both', which='major', labelsize=10)

    labels = [convector.index_to_breed[i] for i in range(10)]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    fig.savefig(f'{save_path}.jpeg', bbox_inches='tight', pad_inches=1)
    plt.close()


def load_model(run, run_name_to_model):
    model = getattr(models, run.config['model_name'])()
    model.classifier[-1] = Linear(model.classifier[-1].in_features, 10)
    restore_model(model, run.id, run_name_to_model)
    model.eval()
    return model


@contextmanager
def stopwatch() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def restore_model(model, run_name, run_name_to_model):
    model_name_file = f'{run_name_to_model[run_name]}.pt'
    local_model_path = join('models', run_name, model_name_file)

    if not exists(local_model_path):
        wandb.restore(model_name_file, f'stasiche/SberCVDS/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
