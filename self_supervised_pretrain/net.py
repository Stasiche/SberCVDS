import torch

import torchvision.models as models
from torch import nn
from torch.nn import Linear


class Net(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=True)
        self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, 10)

    def forward(self, x):
        x = self.model.features(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        clf_out = self.model.classifier(x)
        return clf_out, x
