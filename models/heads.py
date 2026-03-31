"""
Task-Specific Heads

Each head takes the shared neck output (256ch, H/4, W/4) and produces
task-specific predictions. Heads are lightweight by design — most of the
computation happens in the shared backbone + neck.

Heads in this file:
- SegmentationHead: pixel-wise class predictions (Module 2)
- DepthHead: monocular depth estimation (Module 2)
- LanesHead: ego-lane segmentation (Module 3 - student lab)
- DetectionHead: 2D object detection heatmap + regression (Module 3 - student lab)
- TrajectoryHead: steering/trajectory prediction (Module 3 - student lab)

Architecture based on Autoware Vision Pilot:
https://github.com/autowarefoundation/autoware_vision_pilot
"""

import torch
import torch.nn as nn


# =============================================================================
# MODULE 2 HEADS (taught by instructor)
# =============================================================================

class SegmentationHead(nn.Module):
    """Semantic segmentation head.
    Upsamples neck from H/4 to H/1 and predicts per-pixel class labels.

    Args:
        num_classes: number of semantic classes (default=19 for BDD100K/Cityscapes)
    """

    def __init__(self, num_classes=19):
        super(SegmentationHead, self).__init__()
        self.GeLU = nn.GELU()

        # Upsample block 1: H/4 -> H/2
        self.upsample_layer_0 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)  # Match features[0] channels
        self.decode_layer_0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(256, 128, 3, 1, 1)

        # Upsample block 2: H/2 -> H
        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(128, 64, 3, 1, 1)

        # Output
        self.output_layer = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, neck, features):
        """
        Args:
            neck: (B, 256, H/4, W/4) from SceneNeck
            features: encoder features list, uses features[0] for skip
        Returns:
            (B, num_classes, H, W) logits
        """
        # H/4 -> H/2
        d0 = self.upsample_layer_0(neck)
        d0 = d0 + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))

        # H/2 -> H
        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))

        output = self.output_layer(d3)
        return output


class DepthHead(nn.Module):
    """Monocular depth estimation head.
    Same structure as SegmentationHead but outputs 1 channel (depth map).
    """

    def __init__(self):
        super(DepthHead, self).__init__()
        self.GeLU = nn.GELU()

        # Upsample block 1: H/4 -> H/2
        self.upsample_layer_0 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)
        self.decode_layer_0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(256, 128, 3, 1, 1)

        # Upsample block 2: H/2 -> H
        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(128, 128, 3, 1, 1)

        # Output: single channel depth
        self.output_layer = nn.Conv2d(128, 1, 3, 1, 1)

    def forward(self, neck, features):
        """
        Args:
            neck: (B, 256, H/4, W/4) from SceneNeck
            features: encoder features list, uses features[0] for skip
        Returns:
            (B, 1, H, W) depth prediction
        """
        d0 = self.upsample_layer_0(neck)
        d0 = d0 + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))

        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))

        prediction = self.output_layer(d3)
        return prediction


# =============================================================================
# MODULE 3 HEADS (student lab — instructor demos one, students build the rest)
# =============================================================================

class LanesHead(nn.Module):
    """Ego-lane segmentation head.
    Operates at neck resolution (H/4) — no further upsampling needed
    for lane detection since lanes are thin structures.

    Outputs 3 channels: background, left lane, right lane.
    """

    def __init__(self, num_classes=3):
        super(LanesHead, self).__init__()
        self.GeLU = nn.GELU()

        self.decode_layer_0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.output_layer = nn.Conv2d(128, num_classes, 3, 1, 1)

    def forward(self, neck, features=None):
        """
        Args:
            neck: (B, 256, H/4, W/4) from SceneNeck
            features: unused, kept for consistent interface
        Returns:
            (B, num_classes, H/4, W/4) lane logits
        """
        d0 = self.GeLU(self.decode_layer_0(neck))
        d1 = self.GeLU(self.decode_layer_1(d0))
        output = self.output_layer(d1)
        return output


class DetectionHead(nn.Module):
    """Lightweight 2D object detection head (anchor-free, CenterNet-style).
    Predicts heatmaps + bounding box offsets at neck resolution.

    Outputs:
        - heatmap: (B, num_classes, H/4, W/4) — object center probability
        - regression: (B, 4, H/4, W/4) — bbox offset (dx, dy, w, h)
    """

    def __init__(self, num_classes=10):
        super(DetectionHead, self).__init__()
        self.GeLU = nn.GELU()

        # Shared layers
        self.shared_0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.shared_1 = nn.Conv2d(256, 128, 3, 1, 1)

        # Heatmap branch (object centers)
        self.heatmap_layer = nn.Conv2d(128, num_classes, 3, 1, 1)

        # Regression branch (bbox: dx, dy, w, h)
        self.regression_layer = nn.Conv2d(128, 4, 3, 1, 1)

    def forward(self, neck, features=None):
        """
        Args:
            neck: (B, 256, H/4, W/4) from SceneNeck
            features: unused, kept for consistent interface
        Returns:
            heatmap: (B, num_classes, H/4, W/4)
            regression: (B, 4, H/4, W/4)
        """
        d0 = self.GeLU(self.shared_0(neck))
        d1 = self.GeLU(self.shared_1(d0))

        heatmap = torch.sigmoid(self.heatmap_layer(d1))
        regression = self.regression_layer(d1)

        return heatmap, regression


class TrajectoryHead(nn.Module):
    """Trajectory/steering prediction head.
    Takes neck features, pools them down, and predicts a steering angle
    as a classification over discrete bins (following Autoware's approach).

    Outputs steering angle as one of 61 classes: -30 to +30 degrees.
    """

    def __init__(self, num_bins=61):
        super(TrajectoryHead, self).__init__()
        self.GeLU = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((5, 10))
        self.dropout = nn.Dropout(p=0.4)

        # Reduce spatial features
        self.reduce_0 = nn.Conv2d(256, 128, 3, 1, 1)
        self.reduce_1 = nn.Conv2d(128, 64, 3, 1, 1)

        # Classification layers
        self.fc_0 = nn.Linear(64 * 5 * 10, 512)
        self.fc_1 = nn.Linear(512, num_bins)

    def forward(self, neck, features=None):
        """
        Args:
            neck: (B, 256, H/4, W/4) from SceneNeck
            features: unused, kept for consistent interface
        Returns:
            (B, num_bins) steering angle logits
        """
        x = self.pool(neck)
        x = self.GeLU(self.reduce_0(x))
        x = self.GeLU(self.reduce_1(x))

        x = torch.flatten(x, start_dim=1)
        x = self.GeLU(self.dropout(self.fc_0(x)))
        steering = self.fc_1(x)

        return steering
