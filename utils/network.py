from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

class ClassifierBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = 768
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x, y=None, z=None, action=None):
        if action == 'val':
            f = self.backbone(x=x, action='val')
            f = f.view(-1, 768)
            f = self.bottleneck(f)        # 32,256
            predictions = self.head(f)    # 32,31
            return predictions
        elif action == 'memory':
            f1, f2, f3, f4, f = self.backbone(x=x, y=y, action=action)
            fx1, fy1 = f1.chunk(2, dim=0)
            fx2, fy2 = f2.chunk(2, dim=0)
            fx3, fy3 = f3.chunk(2, dim=0)
            fx4, fy4 = f4.chunk(2, dim=0)
            px, py = f.chunk(2, dim=0)

            px = self.bottleneck(px.view(-1, 768))
            py = self.bottleneck(py.view(-1, 768))
            x = self.head(px)
            y = self.head(py)
            return fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, px, py, x, y
        elif action == 'train':
            f1, f2, f3, f4, f = self.backbone(x=x, y=y, z=z, action='train')
            fx1, fy1, fz1 = f1.chunk(3, dim=0)
            fx2, fy2, fz2 = f2.chunk(3, dim=0)
            fx3, fy3, fz3 = f3.chunk(3, dim=0)
            fx4, fy4, fz4 = f4.chunk(3, dim=0)

            px, py, pz = f.chunk(3, dim=0)
            px = self.bottleneck(px.view(-1, 768))
            py = self.bottleneck(py.view(-1, 768))
            pz = self.bottleneck(pz.view(-1, 768))
            x = self.head(px)
            y = self.head(py)   
            z = self.head(pz)

            return fx1, fy1, fz1, fx2, fy2, fz2, fx3, fy3, fz3, fx4, fy4, fz4, px, py, pz, x, y, z



    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(768, bottleneck_dim),      # 2048, 31
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)





