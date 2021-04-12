import torch
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCNHead, FCN


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet18()
    return_layers = {'layer4': '4', 'layer3': '3', 'layer2': '2'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    a = torch.rand(1, 3, 512, 512)
    r = backbone(a)

    for k, v in r.items():
        print(k, v.shape)
