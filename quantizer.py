import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np

CLIPMIN = 1e-5

class Quantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
    ):
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic


        self.mode = "calibration"
        self.enable = True
        self.recorded_quant_input=None

    def fake_quant(self, x, scale, round_zero_point):
        # start quantization
        x_int = (x / scale).round_()
        if round_zero_point is not None:
            x_int = x_int.add_(round_zero_point)
        x_int = x_int.clamp_(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            # print(round_zero_point)
            x_dequant = x_dequant.sub_(round_zero_point)
        x_dequant = x_dequant.mul_(scale)
        return x_dequant

    def calibration(self, x: torch.Tensor):
        reduce_axes = [
            _
            for _ in range(x.ndim)
            if _ not in self.per_channel_axes and _ != self.cluster_dim
        ]
        # print(reduce_axes)

        if self.metric in ["layer_mse", "minmax", "ema_minmax"]:
            scale, zero_point = self.minmax_calibration(x, reduce_axes)


        del self.scale
        self.register_buffer("scale", scale)
        # self.scale = scale
        if zero_point is not None:
            zero_point.clamp_(min=-1e5, max=1e5)
            del self.zero_point, self.round_zero_point
            self.register_buffer("zero_point", zero_point)
            self.register_buffer("round_zero_point", zero_point.round())
            # self.zero_point = zero_point
            # self.round_zero_point = self.zero_point.round()

        # debug
        if not self.dynamic:
            if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
                breakpoint()
            if self.zero_point is not None:
                if (
                    torch.isinf(self.round_zero_point).any()
                    or torch.isnan(self.round_zero_point).any()
                ):
                    breakpoint()

    def minmax_calibration(self, x, reduce_axes):
        # minmax
        if self.symmetric:
            if len(reduce_axes):
                abs_max = x.abs().amax(reduce_axes, keepdim=True)
            else:
                abs_max = x.abs()
            scale = abs_max / (2 ** (self.n_bits - 1))
            if self.cluster_dim is not None:
                # for cluster quantization
                st = 0
                for count in self.cluster_counts:
                    part_scale = torch.narrow(scale, self.cluster_dim, st, count)
                    cluster_max = part_scale.amax(
                        self.cluster_dim, keepdim=True
                    )  # 不该保持维度
                    scale.narrow(self.cluster_dim, st, count).copy_(cluster_max)
                    st += count
            scale.clamp_(min=CLIPMIN, max=1e5)
            zero_point = None
        else:
            if len(reduce_axes):
                xmin = x.amin(reduce_axes, keepdim=True)
                xmax = x.amax(reduce_axes, keepdim=True)
            else:
                xmin = x.clone()
                xmax = x.clone()
            if not self.dynamic:
                if self.cached_xmax is not None:
                    if self.metric == "minmax":
                        xmax = torch.max(self.cached_xmax, xmax)
                        xmin = torch.min(self.cached_xmin, xmin)
                    if self.metric == "ema_minmax":
                        xmax = self.cached_xmax * 0.99 + xmax * 0.01
                        xmin = self.cached_xmin * 0.99 + xmin * 0.01
                self.cached_xmax = xmax
                self.cached_xmin = xmin
            scale = (xmax - xmin) * (2**-self.n_bits)
            scale.clamp_(min=CLIPMIN, max=1e4)
            zero_point = (xmax + xmin) * (-0.5 / scale)
            return scale, zero_point

    def free(self):
        del self.cached_xmin
        del self.cached_xmax
        del self.recorded_quant_input


