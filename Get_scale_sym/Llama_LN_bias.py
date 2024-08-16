import torch
from torch import nn

class LlamaRMSNorm_bias(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return ((self.weight * hidden_states) - self.bias).to(input_dtype)

    @staticmethod
    def from_float(module, input_scales, hidden_size):
        new_module = LlamaRMSNorm_bias(hidden_size)
        new_module.weight = module.weight
        bias = torch.nn.Parameter(input_scales)
        new_module.bias = bias
        return new_module

class LlamaLinear_bias(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.zeros(
            (self.out_features, self.in_features), dtype=torch.float16, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float16, requires_grad=False))


    @torch.no_grad()
    def forward(self, x):
        x_dtype = x.dtype
        # print(self.weight.shape)
        y = torch.matmul(x, self.weight.t())
        y = torch.add(y, self.bias).to(x_dtype)
        return y


    @staticmethod
    def from_float(module: torch.nn.Linear, bias):
        out_features,  in_features= module.weight.size()
        int8_module = LlamaLinear_bias(
            in_features, out_features)
        int8_module.weight = module.weight
        int8_module.bias = bias
        return int8_module