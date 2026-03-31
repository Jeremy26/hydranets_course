"""
HydraNet: Multi-Task Network with Shared Backbone

Combines a shared EfficientNet-B0 backbone with task-specific heads.
The architecture follows the Autoware Vision Pilot pattern:

    Image -> Backbone -> Context -> Neck -> Head(s)

Multiple heads share the same backbone and neck, each producing
different outputs (segmentation, depth, lanes, etc.).

Architecture based on Autoware Vision Pilot:
https://github.com/autowarefoundation/autoware_vision_pilot
"""

import torch.nn as nn

from .backbone import Backbone, FrozenBackbone
from .context import SceneContext, DepthContext
from .neck import SceneNeck
from .heads import SegmentationHead, DepthHead


class HydraNet(nn.Module):
    """Two-task HydraNet for Module 2: Segmentation + Depth.

    Architecture:
        Backbone (shared) -> Context (per-task) -> Neck (shared) -> Heads (per-task)

    Note: Both tasks share the same neck but have separate context modules,
    allowing each task to attend to different scene-level features.
    """

    def __init__(self, num_seg_classes=19):
        super(HydraNet, self).__init__()

        # Shared encoder
        self.backbone = Backbone(pretrained=True)

        # Task-specific context modules
        self.seg_context = SceneContext()
        self.depth_context = DepthContext()

        # Shared decoder neck
        self.neck = SceneNeck()

        # Task-specific heads
        self.seg_head = SegmentationHead(num_classes=num_seg_classes)
        self.depth_head = DepthHead()

    def forward(self, image):
        # Shared feature extraction
        features = self.backbone(image)
        deep_features = features[4]  # (B, 1280, H/32, W/32)

        # Segmentation branch
        seg_context = self.seg_context(deep_features)
        seg_neck = self.neck(seg_context, features)
        seg_output = self.seg_head(seg_neck, features)

        # Depth branch
        depth_context = self.depth_context(deep_features)
        depth_neck = self.neck(depth_context, features)
        depth_output = self.depth_head(depth_neck, features)

        return seg_output, depth_output

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        """Returns parameters for everything except the backbone
        (context, neck, heads) — useful for differential learning rates."""
        params = []
        for module in [self.seg_context, self.depth_context,
                       self.neck, self.seg_head, self.depth_head]:
            params.extend(module.parameters())
        return params


class HydraNetExtended(nn.Module):
    """Extended HydraNet for Module 3: adds new heads on top of a frozen backbone.

    Students use this to add their own heads (lanes, detection, trajectory)
    without retraining the backbone.
    """

    def __init__(self, pretrained_hydranet, new_heads=None):
        super(HydraNetExtended, self).__init__()

        # Freeze the pre-trained backbone
        self.backbone = FrozenBackbone(pretrained_hydranet.backbone)

        # Keep the pre-trained context and neck (also frozen)
        self.neck = pretrained_hydranet.neck
        self.context = pretrained_hydranet.seg_context
        for param in self.neck.parameters():
            param.requires_grad = False
        for param in self.context.parameters():
            param.requires_grad = False

        # New heads (trainable) — students add these
        self.heads = nn.ModuleDict(new_heads or {})

    def add_head(self, name, head):
        """Add a new task head. Only the head parameters are trainable."""
        self.heads[name] = head

    def forward(self, image):
        # Shared frozen feature extraction
        features = self.backbone(image)
        deep_features = features[4]
        context = self.context(deep_features)
        neck = self.neck(context, features)

        # Run all heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(neck, features)

        return outputs

    def get_trainable_params(self):
        """Returns only the new head parameters (for optimizer)."""
        return self.heads.parameters()
