from .har_cnn import har_cnn
from .kws_cnn import KWS_CNN_S
from .resnet import cifar_resnet10, cifar_resnet20, cifar_resnet56
from .vgg import cifar_vgg16_bn, cifar_vgg19_bn

__all__ = [
    "cifar_resnet10",
    "cifar_resnet20",
    "cifar_resnet56",
    "cifar_vgg16_bn",
    "cifar_vgg19_bn",
    "har_cnn",
    "KWS_CNN_S",
]
