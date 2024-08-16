import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
from Llama_LN_bias import LlamaRMSNorm_bias, LlamaLinear_bias

@torch.no_grad()
def symmetrization_ln_fcs(ln, fcs, act_scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not isinstance(act_scales, dict):
        pass
    else:
        act_scales = act_scales.get('input')
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scales = act_scales.to(device=device, dtype=dtype)
    # ln.weight.div_(scales)

    ln.bias.sub_(scales.view(ln.bias.shape))

    for fc in fcs:
        fc.bias.add_(torch.mm(scales.view(1,-1), fc.weight.t()).view(fc.bias.shape))


@torch.no_grad()
def symmetrization_lm(model, scales):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            # print(True)
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']["input"]
            qkv_input_scales = qkv_input_scales/512
            symmetrization_ln_fcs(attn_ln, qkv, qkv_input_scales)
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']["input"]
            fc1_input_scales = fc1_input_scales/512
            symmetrization_ln_fcs(ffn_ln, fc1, fc1_input_scales)

            # print(True)
            # attn_ln = module.self_attn_layer_norm
            # qkv = [module.self_attn.q_proj,
            #        module.self_attn.k_proj, module.self_attn.v_proj]
            # import re
            # first_number = int(re.search(r'\d+', name).group())
            # # print("first", first_number)
            # # print((first_number + 1) * 2 - 2)
            # qkv_input_scales = scales[(first_number + 1) * 2 - 2]
            # # qkv_input_scales = qkv_input_scales/512
            # symmetrization_ln_fcs(attn_ln, qkv, qkv_input_scales)
            #
            # ffn_ln = module.final_layer_norm
            # fc1 = module.fc1
            # fc1_input_scales = scales[((first_number + 1) * 2) - 1]
            # # fc1_input_scales = fc1_input_scales/512
            # symmetrization_ln_fcs(ffn_ln, fc1, fc1_input_scales)


        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']["input"]
            qkv_input_scales = qkv_input_scales / 512
            symmetrization_ln_fcs(attn_ln, qkv, qkv_input_scales)
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']["input"]
            fc1_input_scales = fc1_input_scales / 512
            symmetrization_ln_fcs(ffn_ln, fc1, fc1_input_scales)
        elif isinstance(module, LlamaDecoderLayer):
            device, dtype = module.input_layernorm.weight.device, module.input_layernorm.weight.dtype

            input_gamma = (module.input_layernorm.weight).to(device).to(dtype)
            module.self_attn.q_proj.weight = module.self_attn.q_proj.weight.mul_(input_gamma)
            module.self_attn.k_proj.weight = module.self_attn.k_proj.weight.mul_(input_gamma)
            module.self_attn.v_proj.weight = module.self_attn.v_proj.weight.mul_(input_gamma)
            module.input_layernorm.weight = module.input_layernorm.weight.div_(input_gamma)

            post_gamma = (module.post_attention_layernorm.weight).to(device).to(dtype)
            module.mlp.gate_proj.weight = module.mlp.gate_proj.weight.mul_(post_gamma)
            module.mlp.up_proj.weight = module.mlp.up_proj.weight.mul_(post_gamma)
            module.post_attention_layernorm.weight = module.post_attention_layernorm.weight.div_(post_gamma)

    # 给Llama加偏置，强制进行对称化操作。
        #elif isinstance(module, LlamaDecoderLayer):
        #    hidden_size = module.input_layernorm.weight.size()
        #    qkv_input_scales =  torch.squeeze(scales[name + '.self_attn.q_proj']["input"])
        #    qkv_input_scales = qkv_input_scales / 512
        #    module.input_layernorm = LlamaRMSNorm_bias.from_float(module.input_layernorm, qkv_input_scales, hidden_size)

        #    Q_bias = torch.mm(qkv_input_scales.view(1,-1), module.self_attn.q_proj.weight.t())
        #    K_bias = torch.mm(qkv_input_scales.view(1, -1), module.self_attn.k_proj.weight.t())
        #    V_bias = torch.mm(qkv_input_scales.view(1, -1), module.self_attn.v_proj.weight.t())
        #    module.self_attn.q_proj = LlamaLinear_bias.from_float(module.self_attn.q_proj, Q_bias)
        #    module.self_attn.k_proj = LlamaLinear_bias.from_float(module.self_attn.k_proj, K_bias)
        #    module.self_attn.v_proj = LlamaLinear_bias.from_float(module.self_attn.v_proj, V_bias)


        #    fc1_input_scales = torch.squeeze(scales[name + '.mlp.gate_proj']["input"])
        #    fc1_input_scales = fc1_input_scales / 512
        #    module.post_attention_layernorm = LlamaRMSNorm_bias.from_float(module.post_attention_layernorm, fc1_input_scales, hidden_size)
        #    gate_bias = torch.mm(fc1_input_scales.view(1,-1), module.mlp.gate_proj.weight.t())
        #    up_bias = torch.mm(fc1_input_scales.view(1, -1), module.mlp.up_proj.weight.t())
        #    module.mlp.gate_proj = LlamaLinear_bias.from_float(module.mlp.gate_proj, gate_bias)
        #    module.mlp.up_proj = LlamaLinear_bias.from_float(module.mlp.up_proj, up_bias)
