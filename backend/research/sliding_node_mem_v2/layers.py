"""Same dense causal convolution weights, evaluated as a small-window GEMM."""
import torch
from torch.nn import functional as F
from sliding_node_mem_upgrade.model import TemporalCNN as ReferenceCNN


class TemporalCNN(ReferenceCNN):
    def __init__(self, dim=128, backend="auto"):
        super().__init__(dim)
        if backend not in ("auto", "linear", "conv"):
            raise ValueError("CNN backend must be auto, linear or conv")
        self.backend = backend

    def forward(self, x, valid):
        use_linear = self.backend == "linear" or (self.backend == "auto" and x.device.type == "cpu" and len(x) <= 64)
        if not use_linear:
            return super().forward(x, valid)
        width = x.shape[1]
        for conv, norm in zip(self.convs, self.norms):
            dilation = conv.dilation[0]
            # Conv1d uses [out, in, kernel]; concatenation is [kernel, in].
            patches = torch.cat([F.pad(x, (0, 0, offset, 0))[:, :width]
                                 for offset in (2*dilation, dilation, 0)], -1)
            weight = conv.weight.permute(0, 2, 1).reshape(conv.out_channels, -1)
            y = F.linear(patches, weight, conv.bias)
            x = norm(x + F.gelu(y)) * valid[..., None]
        return x
