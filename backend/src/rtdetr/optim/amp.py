import torch
import torch.nn as nn 
import torch.cuda.amp as amp


from ..core import register
from ..misc.dist as dist 


__all__ = ['GradScaler']

GradScaler = register(amp.grad_scaler.GradScaler)
