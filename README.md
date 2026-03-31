# HydraNet Course

Multi-Task Learning for Autonomous Driving Perception.

Course: https://courses.thinkautonomous.ai/hydranets

## Architecture

Based on the [Autoware Vision Pilot](https://github.com/autowarefoundation/autoware_vision_pilot) HydraNet pattern:

```
                                    ┌─> Segmentation (19 classes)
                                    ├─> Depth (monocular)
Image -> EfficientNet-B0 -> Context -> Neck ├─> Lanes (ego-lane)
                                    ├─> 2D Detection (CenterNet)
                                    └─> Trajectory (steering)
```

## Course Structure

| Module | Topic | Notebook |
|--------|-------|----------|
| **1: MTL Basics** | Multi-task face analysis (age, gender, race) | `MTL_PyTorch.ipynb` |
| **2: Dense Prediction** | Segmentation + Depth on BDD100K | `Module_2_Depth_Segmentation.ipynb` |
| **3: Advanced Heads Lab** | Add new heads (lanes, detection, trajectory) | `Module_3_Advanced_Heads_Lab.ipynb` |

## Key Files

```
models/
  backbone.py       # EfficientNet-B0 encoder
  context.py         # Global scene context (pseudo-attention)
  neck.py            # U-Net decoder with skip connections
  heads.py           # Task-specific heads (seg, depth, lanes, detection, trajectory)
  hydranet.py        # HydraNet and HydraNetExtended wrappers
utils/
  data.py            # BDD100K dataset and dataloaders
  losses.py          # InverseHuber, MultiTaskLoss
  metrics.py         # mIoU, RMSE
```

## Dataset

- **Module 1:** UTKFace (face analysis)
- **Modules 2-3:** [BDD100K](https://www.vis.xyz/bdd100k/) + pseudo-depth from [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)

## Legacy Notebooks

The original course notebooks (MobileNetv2 + RefineNet on NYUDv2) are preserved:
- `MTL_Train_Home.ipynb` / `MTL_Train_Home_S.ipynb`
- `MTL_Run.ipynb` / `MTL_Run_S.ipynb` / `MTL_Run_Py.ipynb` / `MTL_Run_S_Py.ipynb`
