import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# import matplotlib.pyplot as plt


activation_para = 4

@torch.no_grad()
def smooth_ln_fcs(fcs, act_scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not isinstance(act_scales, dict):
        pass
    else:
        act_scales = act_scales.get('input')


    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.clamp(min=1e-5).to(device=device, dtype=dtype)
    # ln.weight.div_(scales)
    # ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(act_scales.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']["input"]
            qkv_input_scales = qkv_input_scales / (2 ** (activation_para - 1) - 1)
            smooth_ln_fcs(qkv, qkv_input_scales)

            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']["input"]
            fc1_input_scales = fc1_input_scales / (2 ** (activation_para - 1) - 1)
            smooth_ln_fcs(fc1, fc1_input_scales)
        elif isinstance(module, BloomBlock):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']["input"]
            qkv_input_scales = qkv_input_scales / (2 ** (activation_para - 1) - 1)
            # print(qkv_input_scales)
            # print(qkv_input_scales.max())
            smooth_ln_fcs(qkv, qkv_input_scales)
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']["input"]
            fc1_input_scales = fc1_input_scales / (2 ** (activation_para - 1) - 1)
            smooth_ln_fcs(fc1, fc1_input_scales)
        elif isinstance(module, LlamaDecoderLayer):
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales =  torch.squeeze(scales[name + '.self_attn.q_proj']["input"])
            qkv_input_scales = qkv_input_scales / (2 ** (activation_para - 1) - 1)
            # print(qkv_input_scales)
            # print(qkv_input_scales.max())
            smooth_ln_fcs(qkv, qkv_input_scales)

            fc1 = [module.mlp.gate_proj, module.mlp.up_proj]
            fc1_input_scales = torch.squeeze(scales[name + '.mlp.gate_proj']["input"])
            fc1_input_scales = fc1_input_scales / (2 ** (activation_para - 1) - 1)
            smooth_ln_fcs(fc1, fc1_input_scales)
