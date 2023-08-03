from distutils.log import error
from functools import partial
from mimetypes import init
from types import new_class
from matplotlib.pyplot import axis
import numpy as np
from sklearn.utils import deprecated
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch import Tensor, ctc_loss
from typing import Callable, Optional

# Note: This sentence would impact the device selection
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class S4L:
    def __init__(self, w=1): 
        self.w = w
    
    # class_logits: ground truth label logits
    # class_labels: ground truth label
    # rot_logits: rotation degree predicted logits
    # rot_labels: ground truth rotation degree 
    def loss(self, class_logits, class_labels, rot_logits, rot_labels):
        sup_loss, rot_loss = 0,0
        if class_logits is not None and class_labels is not None:
            sup_loss = F.cross_entropy(class_logits, class_labels, reduction="mean")
            self.sup_loss = sup_loss.detach().cpu().numpy()
        if rot_logits is not None and rot_labels is not None:
            rot_loss = F.cross_entropy(rot_logits, rot_labels, reduction="mean")
            self.rot_loss = rot_loss.detach().cpu().numpy()
            
        return sup_loss + self.w * rot_loss

    