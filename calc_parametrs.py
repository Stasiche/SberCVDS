import torchvision.models as models
from torch.nn import Linear

for model_name in ['convnext_tiny', 'convnext_small',
                   'efficientnet_b0', 'efficientnet_b5',
                   'vgg11_bn', 'vgg16_bn',
                   'vit_b_16', 'vit_b_32',
                   'regnet_y_32gf']:
    m = getattr(models, model_name)()
    if hasattr(m, 'classifier'):
        m.classifier[-1] = Linear(m.classifier[-1].in_features, 10)
    elif hasattr(m, 'heads'):
        m.heads[0] = Linear(m.heads[0].in_features, 10)
    else:
        m.fc = Linear(m.fc.in_features, 10)
    print(model_name, str(round(sum(p.numel() for p in m.parameters())/1e6, 1))+'M')