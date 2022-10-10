import logging

import torch

logger = logging.getLogger(__name__)

model_name_list: list = [
    'vgg16', 'mobilenetv2'
]


def get_model(model_name: str = 'mobilenetv2'):
    if model_name not in model_name_list:
        raise ValueError
    match model_name:
        case 'vgg16':
            """ref: https://pytorch.org/hub/pytorch_vision_vgg/"""
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
            return model.eval()

        case 'mobilenetv2':
            """ref: https://zenn.dev/kmiura55/articles/pytorch-use-mobilenetv2"""
            from torchvision.models import mobilenetv2
            Model = mobilenetv2.mobilenet_v2(pretrained=True)
            return Model.eval()

        case _:
            logger.info(f'Not Set Model: {model_name}')
