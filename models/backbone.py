"""
Backbone: EfficientNet-B0 Encoder

Extracts multi-scale features from input images using a pretrained EfficientNet-B0.
Returns features at 5 different scales for use by the decoder neck and heads.

Architecture based on Autoware Vision Pilot:
https://github.com/autowarefoundation/autoware_vision_pilot
"""

import torch.nn as nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        if pretrained:
            self.encoder = models.efficientnet_b0(
                weights='EfficientNet_B0_Weights.IMAGENET1K_V1'
            ).features
        else:
            self.encoder = models.efficientnet_b0(weights=None).features

    def forward(self, image):
        # EfficientNet-B0 feature stages
        # Input: (B, 3, H, W)
        l0 = self.encoder[0](image)   # (B, 32, H/2, W/2)    - stride 2
        l1 = self.encoder[1](l0)      # (B, 16, H/2, W/2)    - stride 2
        l2 = self.encoder[2](l1)      # (B, 24, H/4, W/4)    - stride 4
        l3 = self.encoder[3](l2)      # (B, 40, H/8, W/8)    - stride 8
        l4 = self.encoder[4](l3)      # (B, 80, H/16, W/16)  - stride 16
        l5 = self.encoder[5](l4)      # (B, 112, H/16, W/16) - stride 16
        l6 = self.encoder[6](l5)      # (B, 192, H/32, W/32) - stride 32
        l7 = self.encoder[7](l6)      # (B, 320, H/32, W/32) - stride 32
        l8 = self.encoder[8](l7)      # (B, 1280, H/32, W/32)- stride 32

        # Return multi-scale features for skip connections
        # features[0] = low-level (high res), features[4] = deep (low res)
        return [l0, l2, l3, l4, l8]


class FrozenBackbone(nn.Module):
    """Wraps a trained backbone and freezes all parameters.
    Used in Module 3 when students add new heads to a pre-trained model."""

    def __init__(self, pretrained_model):
        super(FrozenBackbone, self).__init__()
        self.backbone = pretrained_model
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, image):
        return self.backbone(image)
