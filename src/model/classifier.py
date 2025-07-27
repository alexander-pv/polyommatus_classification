import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import ResNet50_Weights
from torchvision.models import efficientnet_v2_l
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet50
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

MODELS = {
    "resnet50",
    "efficientnet_v2_l",
    "mobilenet_v3_large",
}
MODELS_HEAD_PARAMS = {
    "resnet50": {"fc.weight", "fc.bias"},
    "efficientnet_v2_l": {"classifier.1.weight", "classifier.1.bias"},
    "mobilenet_v3_large": {
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    },
}


def get_pretrained_model(
    name: str, n_classes: int, freeze: bool = False
) -> torch.nn.Module:
    """
    Prepare pretrained torch model with custom number of classes

    Layers that are replaced:
    resnet50:
        (fc): Linear(in_features=2048, out_features=1000, bias=True)
    efficientnet_v2_l:
        (classifier): Sequential(
                                (0): Dropout(p=0.4, inplace=True)
                                (1): Linear(in_features=1280, out_features=1000, bias=True)
                                )
    mobilenet_v3_large:
        (classifier): Sequential(
                                (0): Linear(in_features=960, out_features=1280, bias=True)
                                (1): Hardswish()
                                (2): Dropout(p=0.2, inplace=True)
                                (3): Linear(in_features=1280, out_features=1000, bias=True)
                                )
                                )

    :param name:        Model name
    :param n_classes:   The number of classes
    :param freeze:      Whether to freeze the model (feature extractor) except the last linear layer
    :return:            torch model with custom number of classes
    """
    logger.info(f"Loading {name} model. Number of classes: {n_classes}")
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    elif name == "efficientnet_v2_l":
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            *[
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features=n_classes, bias=True),
            ]
        )
    elif name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            *[
                nn.Linear(in_features=960, out_features=1280, bias=True),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=n_classes, bias=True),
            ]
        )
    else:
        err_msg = f"Unknown model: {name}"
        logger.error(err_msg)
        raise NotImplementedError(err_msg)

    unfreeze_model_params = MODELS_HEAD_PARAMS[name]
    if freeze:
        logger.info(f"Freezing {name} params")
        for name, param in model.named_parameters():
            if name not in unfreeze_model_params:
                param.requires_grad = False
            else:
                logger.info(
                    f"This param was not frozen: {name}. Reason: selected for train"
                )
    return model
