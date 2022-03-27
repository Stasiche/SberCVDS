from telegram import Update
from telegram.ext import CallbackContext
from PIL import Image
import io
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import logging
import torch
from torch import nn
from src.breed_convector import BreedConvector

logger = logging.getLogger(__name__)


def recognize(update: Update, context: CallbackContext, model: nn.Module, convector: BreedConvector, img):
    try:
        device = next(model.parameters()).device
        trans = Compose([Resize((256, 256)),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ])

        image = Image.open(io.BytesIO(img))
        image = image.convert('RGB')
        image = trans(image)
        prediction_cls = torch.argmax(model(image.to(device).unsqueeze(0)), dim=1).item()
        update.message.reply_text("It's a " + convector.index_to_breed[prediction_cls])
    except Exception as e:
        logger.error('Error: ' + str(e) +
                     '\nContext: ' + str(context) +
                     '\nUpdate: ' + str(update))
