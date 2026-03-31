"""
Neck: U-Net Style Decoder with Skip Connections

The neck progressively upsamples deep features back to higher resolution,
adding skip connections from the encoder at each stage.

Flow: context (1280ch, H/32) -> ... -> neck output (256ch, H/4)
Each upsample block: ConvTranspose2d -> add skip -> double conv

Architecture based on Autoware Vision Pilot:
https://github.com/autowarefoundation/autoware_vision_pilot
"""

import torch.nn as nn


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
