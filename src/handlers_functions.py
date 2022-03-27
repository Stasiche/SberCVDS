from telegram import Update
from telegram.ext import CallbackContext
import logging
from torch import nn
from src.breed_convector import BreedConvector
from src.utils import recognize

logger = logging.getLogger(__name__)


def recognize_document(update: Update, context: CallbackContext, model: nn.Module, convector: BreedConvector):
    try:
        img = context.bot.get_file(update.message.document).download_as_bytearray()
        recognize(update, context, model, convector, img)
    except Exception as e:
        logger.error('Error: ' + str(e) +
                     '\nContext: ' + str(context) +
                     '\nUpdate: ' + str(update))


def recognize_image(update: Update, context: CallbackContext, model: nn.Module, convector: BreedConvector):
    try:
        img = context.bot.get_file(update.message.photo[-1]).download_as_bytearray()
        recognize(update, context, model, convector, img)
    except Exception as e:
        logger.error('Error: ' + str(e) +
                     '\nContext: ' + str(context) +
                     '\nUpdate: ' + str(update))


def help(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Send me a photo and I will try to determine the breed.")
