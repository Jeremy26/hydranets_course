"""
Context Modules: Global Scene Understanding

Context modules provide global scene understanding by:
1. Global Average Pooling the deep features into a vector
2. Processing through an MLP to capture scene-level information
3. Reshaping back to spatial dimensions
4. Applying pseudo-attention: context * features + features (residual)

This helps the network understand "what kind of scene am I looking at?"
before decoding per-pixel predictions.

Architecture based on Autoware Vision Pilot:
https://github.com/autowarefoundation/autoware_vision_pilot
"""

import torch
import torch.nn as nn


class SceneContext(nn.Module):
    """Context module for scene segmentation.
    Processes 1280-channel deep features from EfficientNet-B0."""

    def __init__(self):
        super(SceneContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        # MLP: compress deep features to a compact representation
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 200)

        # Spatial reconstruction: expand back to feature map
        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        # Global Average Pooling: (B, 1280, H, W) -> (B, 1280)
        feature_vector = torch.mean(features, dim=[2, 3])

        # MLP to capture scene-level context
        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))

        # Reshape to spatial: (B, 200) -> (B, 1, 10, 20)
        b = c2.shape[0]
        c3 = c2.view(b, 1, 10, 20)

        # Reconstruct to match deep feature dimensions
        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))

        # Pseudo-attention: multiply + residual
        context = c7 * features + features
        return context


class DepthContext(nn.Module):
    """Context module for depth estimation.
    Same structure as SceneContext — depth and segmentation
    benefit from different learned context representations."""

    def __init__(self):
        super(DepthContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 200)

        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        feature_vector = torch.mean(features, dim=[2, 3])

        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))

        b = c2.shape[0]
        c3 = c2.view(b, 1, 10, 20)

        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))

        context = c7 * features + features
        return context
