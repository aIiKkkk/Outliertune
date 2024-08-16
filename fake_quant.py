import torch
from torch import nn
from functools import partial
from quantizer import Quantizer

weight_para = 8
activation_para = 8

weight_quant_params = {
        "n_bits": weight_para,
        "per_channel_axes": [0],
        "symmetric": False,
        "metric": "minmax",
    }

def quantize_weight_per_channel_absmax(w, n_bits=8):
    maxq = torch.tensor(2 ** n_bits - 1)
    xmin = w.amin(dim=-1, keepdim=True)
    xmin = torch.minimum(xmin, torch.zeros(xmin.shape, device=xmin.device, dtype=torch.float16))
    xmax = w.amax(dim=-1, keepdim=True)
    xmax = torch.maximum(xmax, torch.zeros(xmin.shape, device=xmin.device, dtype=torch.float16))
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale = (xmax - xmin) / maxq
    zero_point = torch.round(-xmin / scale)
    zero_point = torch.clamp(zero_point, 0, maxq)
    w.div_(scale).round_().add_(zero_point).clamp_(0, maxq).sub_(zero_point).mul_(scale)

    # q = torch.clamp(torch.round(w / scale) + zero_point, 0, maxq)
    # q = scale * (q - zero_point)
    return w

def quantize_weight_per_channel_absmax_f(w, n_bits=8):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

def auto_clip_tensor(x, n_bits=8, n_grid=5, max_shrink=2):
    min_errs = 1
    org_max_val = x.abs().max()
    best_org_max_val = org_max_val
    i = 0
    for i_s in range(int(max_shrink * n_grid)):
        org_max_val = org_max_val * (1 - 0.025 * (i_s / n_grid))
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -(2 ** (n_bits - 1))
        scale = org_max_val.clamp(min=1e-5).div(qmax)
        quantized_w = torch.round(x.div(scale)).clamp(qmin, qmax).mul(scale)
        err = (quantized_w - x).pow(2).mean()
        i = i + 1
        if err < min_errs:
            min_errs = err
            best_org_max_val = org_max_val


    del org_max_val
    del quantized_w
    torch.cuda.empty_cache()
    # print(i)
    return best_org_max_val

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # scales = auto_clip_tensor(w, n_bits=8)
    scales = w.abs().max()
    qmin = -(2 ** (n_bits - 1))
    qmax = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(qmax)
    w.div_(scales).round_().clamp_(qmin, qmax).mul_(scales)
    return w

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_channel_absmax_NoDequant(t, input_scales, n_bits=8):
    input_scales = input_scales.to(t.device)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2**(n_bits-1)-1
    # print(input_scales)
    scales = input_scales / (q_max)
    # print(input_scales.shape)
    #print(input_scales)
    s = (t / (scales.clamp_(min=1e-5))).round_() # .mul_(scales)
    return s


class W8A8Linear_QKVFc1(nn.Module):
    def __init__(self, input_scales, output_scales, in_features, out_features, bias=True, act_quant='per_channel', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scales = input_scales
        self.output_scales = output_scales
        self.weight_quantizer = Quantizer(**weight_quant_params)

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=activation_para)
        elif act_quant == 'per_channel':
            self.act_quant = partial(
                quantize_activation_per_channel_absmax_NoDequant, n_bits=activation_para)
        elif act_quant == 'per_tensor':
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=activation_para)
        else:
            raise ValueError(f'Invalid weight_quant: {act_quant}')

        if quantize_output:
            self.output_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=activation_para)
        else:
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear_QKVFc1, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x, self.input_scales)
        # print(self.weight.shape)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, input_scales = None, output_scales = None, weight_quant='per_channel', act_quant='per_channel', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear_QKVFc1(input_scales, output_scales,
            module.in_features, module.out_features, module.bias is not None,
            act_quant=act_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            w = module.weight.detach()
            new_module.weight = quantize_weight_per_channel_absmax(
                w, n_bits=weight_para)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=weight_para)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module



class W8A8Linear_OFc2(nn.Module):
    def __init__(self, input_scales, in_features, out_features, bias=True, act_quant='per_channel', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scales = input_scales
        self.weight_quantizer = Quantizer(**weight_quant_params)

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=activation_para)
        elif act_quant == 'per_channel':
            self.act_quant_name = 'per_channel'
            self.act_quant = partial(
                quantize_activation_per_channel_absmax_NoDequant, n_bits=activation_para)
        elif act_quant == 'per_tensor':
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=activation_para)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            # self.output_quant_name = self.act_quant_name
            self.output_quant = partial(
                quantize_activation_per_token_absmax, n_bits=activation_para)
        else:
            # self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear_OFc2, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, input_scales = None, weight_quant='per_channel', act_quant='per_channel', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear_OFc2(input_scales,
            module.in_features, module.out_features, module.bias is not None,
            act_quant=act_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            w = module.weight.detach()
            new_module.weight = quantize_weight_per_channel_absmax(
                w, n_bits=weight_para)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=weight_para)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module
