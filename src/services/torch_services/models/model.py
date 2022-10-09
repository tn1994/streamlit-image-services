import torch


def get_model(model_name: str = 'mobilenetv2'):
    if 'vgg16' == model_name:
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

    elif 'mobilenetv2' == model_name:
        """ref: https://zenn.dev/kmiura55/articles/pytorch-use-mobilenetv2"""
        from torchvision.models import mobilenetv2
        Model = mobilenetv2.mobilenet_v2(pretrained=True)
        return Model.eval()