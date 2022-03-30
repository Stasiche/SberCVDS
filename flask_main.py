import torch
from torch.nn import Linear

from flask import Flask, render_template, request
import PIL
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from os.path import join

from src.breed_convector import BreedConvector
import torchvision.models as models
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

local_model_path = join('src', 'models', f'{config["model_name"]}.pt')
app = Flask(__name__, template_folder='src/templates')
model = getattr(models, config['model_arch_name'])(pretrained=False)
model.classifier[-1] = Linear(model.classifier[-1].in_features, 10)
model.load_state_dict(torch.load(local_model_path,  map_location=torch.device('cpu')))
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
    prediction_cls = torch.argmax(model(image.unsqueeze(0)), dim=1).item()

    return "It's a " + convector.index_to_breed[prediction_cls]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
