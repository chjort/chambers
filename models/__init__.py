from .classification.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200
from .segmentation.unet import UNet, UNet_large, UNet_small
from .segmentation.frrn import FRRN_A, FRRN_B
from .segmentation.resnet_fpn import ResNet50_FPN, ResNet101_FPN, ResNet152_FPN
from .segmentation.deeplab import Deeplabv3
from .gan.pix2pix import Pix2Pix
from .builder import get_model
