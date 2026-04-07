"""
HydraNet Architecture — Shared module for Modules 2 and 3.

All architecture classes in one file so notebooks can do:
    from hydranet import HydraNet, Backbone, SceneNeck, ...
"""

import torch
import torch.nn as nn
from torchvision import models

INPUT_SIZE = (320, 640)  # (H, W)


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
        # EfficientNet-B0 feature stages — strides shown for INPUT_SIZE=(256,512)
        l0 = self.encoder[0](image)   # (B,  32, H/2,  W/2)   = (B,  32, 128, 256)
        l1 = self.encoder[1](l0)      # (B,  16, H/2,  W/2)
        l2 = self.encoder[2](l1)      # (B,  24, H/4,  W/4)   = (B,  24,  64, 128)
        l3 = self.encoder[3](l2)      # (B,  40, H/8,  W/8)   = (B,  40,  32,  64)
        l4 = self.encoder[4](l3)      # (B,  80, H/16, W/16)  = (B,  80,  16,  32)
        l5 = self.encoder[5](l4)      # (B, 112, H/16, W/16)
        l6 = self.encoder[6](l5)      # (B, 192, H/32, W/32)  = (B, 192,   8,  16)
        l7 = self.encoder[7](l6)      # (B, 320, H/32, W/32)
        l8 = self.encoder[8](l7)      # (B,1280, H/32, W/32)  = (B,1280,   8,  16)
        return [l0, l2, l3, l4, l8]


class SceneContext(nn.Module):
    """Context module — compresses deep features, reconstructs as pseudo-attention.

    The Linear bottleneck outputs H/32 * W/32 values (computed from INPUT_SIZE),
    which are reshaped back to match the feature map spatial size dynamically.
    """

    def __init__(self):
        super(SceneContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        # MLP: 1280 -> 800 -> 800 -> H/32*W/32 (dynamic from INPUT_SIZE)
        spatial_size = (INPUT_SIZE[0] // 32) * (INPUT_SIZE[1] // 32)
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, spatial_size)

        # Spatial reconstruction: (B,1,H/32,W/32) -> (B,1280,H/32,W/32)
        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        b, c, h, w = features.shape                          # e.g. (B,1280,8,16)
        feature_vector = torch.mean(features, dim=[2, 3])    # (B,1280)
        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))
        c3 = c2.view(b, 1, h, w)                            # (B,1,8,16) — dynamic!
        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))
        return c7 * features + features                      # pseudo-attention


class DepthContext(nn.Module):
    """Context module for depth estimation — same structure as SceneContext."""

    def __init__(self):
        super(DepthContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        spatial_size = (INPUT_SIZE[0] // 32) * (INPUT_SIZE[1] // 32)
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, spatial_size)

        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        b, c, h, w = features.shape
        feature_vector = torch.mean(features, dim=[2, 3])
        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))
        c3 = c2.view(b, 1, h, w)
        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))
        return c7 * features + features


class SceneNeck(nn.Module):
    def __init__(self):
        super(SceneNeck, self).__init__()
        self.GeLU = nn.GELU()

        # Upsample block 1: H/32 -> H/16
        self.upsample_layer_0 = nn.ConvTranspose2d(1280, 1280, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(80, 1280, 1)  # Match features[3] channels
        self.decode_layer_0 = nn.Conv2d(1280, 768, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(768, 768, 3, 1, 1)

        # Upsample block 2: H/16 -> H/8
        self.upsample_layer_1 = nn.ConvTranspose2d(768, 768, 2, 2)
        self.skip_link_layer_1 = nn.Conv2d(40, 768, 1)   # Match features[2] channels
        self.decode_layer_2 = nn.Conv2d(768, 512, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(512, 512, 3, 1, 1)

        # Upsample block 3: H/8 -> H/4
        self.upsample_layer_2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.skip_link_layer_2 = nn.Conv2d(24, 512, 1)   # Match features[1] channels
        self.decode_layer_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.decode_layer_5 = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, context, features):
        """
        Args:
            context: (B, 1280, H/32, W/32) from context module
            features: list of encoder features [l0, l2, l3, l4, l8]
                      features[1]=(B,24,H/4), features[2]=(B,40,H/8),
                      features[3]=(B,80,H/16)
        Returns:
            neck: (B, 256, H/4, W/4)
        """
        # Block 1: H/32 -> H/16
        d0 = self.upsample_layer_0(context)
        d0 = d0 + self.skip_link_layer_0(features[3])
        d1 = self.GeLU(self.decode_layer_0(d0))
        d2 = self.GeLU(self.decode_layer_1(d1))

        # Block 2: H/16 -> H/8
        d3 = self.upsample_layer_1(d2)
        d3 = d3 + self.skip_link_layer_1(features[2])
        d3 = self.GeLU(self.decode_layer_2(d3))
        d4 = self.GeLU(self.decode_layer_3(d3))

        # Block 3: H/8 -> H/4
        d5 = self.upsample_layer_2(d4)
        d5 = d5 + self.skip_link_layer_2(features[1])
        d5 = self.GeLU(self.decode_layer_4(d5))
        neck = self.GeLU(self.decode_layer_5(d5))

        return neck


class DrivableAreaHead(nn.Module):
    """Drivable area segmentation head.
    Upsamples neck from H/4 to full resolution, predicts 3-class drivable mask.

    Classes: 0=background, 1=direct (ego lane), 2=alternative
    """

    def __init__(self, num_classes=3):
        super(DrivableAreaHead, self).__init__()
        self.GeLU = nn.GELU()

        # H/4 -> H/2
        self.upsample_layer_0  = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)
        self.decode_layer_0    = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1    = nn.Conv2d(256, 128, 3, 1, 1)

        # H/2 -> H
        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2   = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3   = nn.Conv2d(128, 64, 3, 1, 1)

        self.output_layer = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, neck, features):
        d0 = self.upsample_layer_0(neck) + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))
        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))
        return self.output_layer(d3)


class DepthHead(nn.Module):
    """Monocular depth estimation head. Outputs single-channel depth map."""

    def __init__(self):
        super(DepthHead, self).__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_0  = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)
        self.decode_layer_0    = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1    = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2   = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3   = nn.Conv2d(128, 128, 3, 1, 1)

        self.output_layer = nn.Conv2d(128, 1, 3, 1, 1)

    def forward(self, neck, features):
        d0 = self.upsample_layer_0(neck) + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))
        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))
        return self.output_layer(d3)


class HydraNet(nn.Module):
    """Two-task HydraNet: Drivable Area + Depth.

    Backbone (shared) -> Context (per-task) -> Neck (shared) -> Heads (per-task)
    """

    def __init__(self, num_drivable_classes=3):
        super(HydraNet, self).__init__()
        self.backbone       = Backbone(pretrained=True)
        self.drivable_context = SceneContext()
        self.depth_context    = DepthContext()
        self.neck             = SceneNeck()
        self.drivable_head    = DrivableAreaHead(num_classes=num_drivable_classes)
        self.depth_head       = DepthHead()

    def forward(self, image):
        features     = self.backbone(image)
        deep         = features[4]

        driv_ctx     = self.drivable_context(deep)
        driv_neck    = self.neck(driv_ctx, features)
        drivable_out = self.drivable_head(driv_neck, features)

        depth_ctx    = self.depth_context(deep)
        depth_neck   = self.neck(depth_ctx, features)
        depth_out    = self.depth_head(depth_neck, features)

        return drivable_out, depth_out

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        params = []
        for m in [self.drivable_context, self.depth_context,
                  self.neck, self.drivable_head, self.depth_head]:
            params.extend(m.parameters())
        return params


