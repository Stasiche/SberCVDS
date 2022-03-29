import os

import wandb
from os.path import join, getsize
from os import makedirs

from src.dataset import ImageWoofDataset
from torch.utils.data import DataLoader

import numpy as np

from src.breed_convector import BreedConvector
from src.utils import load_model, stopwatch, eval_model, calc_parameters, plot_errors
import csv

api = wandb.Api()
convector = BreedConvector()
val_dataset = ImageWoofDataset('val')
valdata = DataLoader(val_dataset, batch_size=50, shuffle=False)
os.makedirs('errors', exist_ok=True)

runs_list = ['3h4ou7sh', '3bfpypi0', '116xetcy']

with open('analyze.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['model_name', 'parameters_number(M)', 'model_size(MB)', 'accuracy', 'time(s)', 'rps'])
    for run_name in runs_list:
        run = api.run(f'stasiche/SberCVDS/{run_name}')
        model = load_model(run)
        with stopwatch() as t:
            acc, answers = eval_model(model, valdata)
            eval_time = t()
        params_num = calc_parameters(model)
        model_size = round(getsize(join('models', run_name, 'model.pt'))/1e6, 1)
        writer.writerow([run.config["model_name"], round(params_num/1e6, 1), model_size, round(acc, 4),
                         round(eval_time, 0),  round(len(valdata) / eval_time, 3)])

        makedirs(join('errors', run.config['model_name']), exist_ok=True)
        base_path = valdata.dataset.datapath
        for i, indx in enumerate(np.where(answers[0] != answers[1])[0]):
            pred, gt = answers[:, indx]
            img_path = valdata.dataset.data[indx][0]

            plot_errors(pred, gt, join(base_path, img_path), join('errors', run.config["model_name"], str(i)), convector)
