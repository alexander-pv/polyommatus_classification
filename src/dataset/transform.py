import albumentations as A
import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights


def get_cls_pretrained_transform(model_name: str) -> transforms.Compose:
    """
    Image transformation pipeline for specific pretrained classification model
    :param model_name:
    :return: transformation pipeline
    """
    pretrained_transform = {
        "resnet50": ResNet50_Weights.DEFAULT.transforms,
        "efficientnet_v2_l": EfficientNet_V2_L_Weights.DEFAULT.transforms,
        "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT.transforms,
    }[model_name]()
    return pretrained_transform


def get_cls_augmentation_transform() -> A.Compose:
    """
    Augmentation transform for model training.
    :return: transformation pipeline
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                p=0.5,
                limit=(-90, 90),
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.ColorJitter(p=0.5),
            A.ChromaticAberration(p=0.5),
        ]
    )
