import torch
from torch import nn

@torch.no_grad()
def quantize_activation_per_channel_absmax_NoDequant(t, extended_scales, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2**(n_bits-1)-1
    # s = torch.round(t / (extended_scales.clamp_(min=1e-5)))
    s = torch.clamp(torch.round(t / (extended_scales.clamp_(min=1e-5))), -(q_max + 1), q_max)
    return s

class OPT_LayerNorm(nn.Module):
    def __init__(self, ori_layer_norm, activation_para, input_scales):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.ori_layer_norm = ori_layer_norm
        self.activation_para = activation_para
        self.input_scales = input_scales
        self.dev = ori_layer_norm.weight.device

    def forward(self, hidden_states):
        output = self.ori_layer_norm.forward(hidden_states.to(self.dev))
        # print("out", output.abs().max(), output.abs().min())
        # print("scale", self.input_scales.max() * (2 ** (self.activation_para - 1) - 1), self.input_scales.min() * (2 ** (self.activation_para - 1) - 1))
        output = quantize_activation_per_channel_absmax_NoDequant(output, self.input_scales.to(self.dev), self.activation_para)
        return output

    @staticmethod
    def from_float(module, input_scales, activation_para):
        new_module = OPT_LayerNorm(module, activation_para, input_scales)
        return new_module
