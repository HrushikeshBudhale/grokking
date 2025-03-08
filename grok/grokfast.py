from typing import Dict, Optional
import torch.nn as nn
from torch import Tensor

def gradfilter_ema(m: nn.Module,
                    grads: Optional[Dict[str, Tensor]] = None,
                    alpha: float = 0.98,
                    lamb: float = 2.0) -> Dict[str, Tensor]:
    
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + (lamb * grads[n])

    return grads