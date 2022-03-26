from efficientnet_pytorch import EfficientNet
import torch
from src.utils import restore_model

from flask import Flask, render_template, request
import PIL
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import sys

sys.path.extend('../src')
from src.breed_convector import BreedConvector

app = Flask(__name__)
run_name = '31dmvl5j'
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10).to('cuda')
restore_model(model, run_name)
model.eval()
convector = BreedConvector()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def predict():
    trans = Compose([Resize((256, 256)),
                     ToTensor(),
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ])
    f = request.files['img']
    image = PIL.Image.open(f)
    image = image.convert('RGB')
    image = trans(image)
    prediction_cls = torch.argmax(model(image.to('cuda').unsqueeze(0)), dim=1).item()

    return "It's a " + convector.index_to_breed[prediction_cls]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
