import logging
import os
from functools import partial

from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

import torch
from torch import nn
import torchvision.models as models

from src.breed_convector import BreedConvector
from src.handlers_functions import recognize_document, recognize_image, help
from os.path import join
import yaml

TOKEN = os.environ.get('TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(model: nn.Module, convector: BreedConvector) -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(MessageHandler(~Filters.command & Filters.photo,
                                          partial(recognize_image, model=model, convector=convector)))

    dispatcher.add_handler(MessageHandler(~Filters.command & Filters.document,
                                          partial(recognize_document, model=model, convector=convector)))

    dispatcher.add_handler(CommandHandler('help', help))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        local_model_path = join('src', 'models', f'{config["model_name"]}.pt')
        model = getattr(models, config['model_arch_name'])(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
        model.eval()
        convector = BreedConvector()
    except Exception as e:
        logger.error(str(e))

    main(model, convector)
