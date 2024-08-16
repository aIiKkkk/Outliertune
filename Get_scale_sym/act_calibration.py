import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict
from Llama_LN_bias import LlamaLinear_bias
from functools import partial
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=4):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

nbit = 6
@torch.no_grad()
def get_static_decoder_layer_scales(model,
                                    tokenizer,
                                    dataset_path,
                                    num_samples=512,
                                    seq_len=512,
                                    ):
    model.eval()
    device = next(model.parameters()).device
    # print(model)
    act_dict = defaultdict(dict)

    def stat_io_hook_token(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            tmp = torch.tensor(0)
            xmin = torch.minimum(x.amin(-1), tmp).view(-1, 1)
            xmax = torch.maximum(x.amax(-1), tmp).view(-1, 1)
            xmax = (torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5))

            q_max = torch.tensor(2 ** (nbit - 1) - 1).to(x.device)
            p_best = torch.tensor(1)
            y_x = torch.matmul(x, m.weight.t())
            mse = True
            if mse:
                best = torch.tensor(1e6)
                for i in range(int(0.5 * 100)):
                    p = 1 - i / 100
                    xmax1 = p * xmax  # * B_mapped
                    scale1 = xmax1 / q_max
                    q = torch.clamp(torch.round(x / (scale1)), -(q_max + 1), q_max).mul(scale1)
                    y_q = torch.matmul(q, m.weight.t())
                    absolute_errors = (torch.abs(y_q - y_x)).pow(2.4)
                    err = torch.sum(absolute_errors, -1)
                    sorted_err, indices = torch.sort(err)
                    if sorted_err.dim() == 1:
                        sorted_err = sorted_err.unsqueeze(0)
                    if sorted_err[:, -1] > 20:
                        n = 3
                    else:
                        n = 0
                    num_to_keep = len(err) - n
                    trimmed_err = sorted_err[:, :num_to_keep]
                    mean_err = torch.mean(trimmed_err)
                    if mean_err < best:
                        best = mean_err
                        p_best = torch.tensor(p)
            act_dict[name]["input"] = p_best
        else:
            tmp = torch.tensor(0)
            xmin = torch.minimum(x.amin(-1), tmp).view(-1, 1)
            xmax = torch.maximum(x.amax(-1), tmp).view(-1, 1)
            xmax = (torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5))

            q_max = torch.tensor(2 ** (nbit - 1) - 1).to(x.device)
            p_best = torch.tensor(1)
            y_x = torch.matmul(x, m.weight.t())
            mse = True
            if mse:
                best = torch.tensor(1e6)
                for i in range(int(0.5 * 100)):
                    p = 1 - i / 100
                    xmax1 = p * xmax  # * B_mapped
                    scale1 = xmax1 / q_max
                    q = torch.clamp(torch.round(x / (scale1)), -(q_max + 1), q_max).mul(scale1)
                    y_q = torch.matmul(q, m.weight.t())
                    absolute_errors = (torch.abs(y_q - y_x)).pow(2.4)
                    err = torch.sum(absolute_errors, -1)
                    sorted_err, indices = torch.sort(err)
                    if sorted_err.dim() == 1:
                        sorted_err = sorted_err.unsqueeze(0)
                    if sorted_err[:, -1] > 20:
                        n = 3
                    else:
                        n = 0
                    num_to_keep = len(err) - n
                    trimmed_err = sorted_err[:, :num_to_keep]
                    mean_err = torch.mean(trimmed_err)
                    if mean_err < best:
                        best = mean_err
                        p_best = torch.tensor(p)
            # act_dict[name]["input"] = torch.max(act_dict[name]["input"], p_best)
            if act_dict[name]["input"] < p_best:
                act_dict[name]["input"] = act_dict[name]["input"] + (p_best * 0.01)
            else:
                act_dict[name]["input"] = act_dict[name]["input"]
            # print(act_dict[name]["input"])

    def stat_io_hook_f(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        # print(x.shape)
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().amax(-2, keepdim=True)
        else:
            new_max_values = x.detach().abs().amax(-2, keepdim=True)
            act_dict[name]["input"] = torch.max(act_dict[name]["input"], new_max_values)


    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            tmp = torch.zeros(x.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0)
            xmin = torch.minimum(x.amin(-2, keepdim=True), tmp)
            xmax = torch.maximum(x.amax(-2, keepdim=True), tmp)
            xmax = (torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5))
            mse = True
            if mse:
                best = torch.full([x.shape[-1]], float('inf'), device=x.device).unsqueeze(0).unsqueeze(0)
                for i in range(int(0.8 * 100)):
                    p = 1 - i / 100
                    xmax1 = p * xmax
                    scale1 = xmax1 / torch.tensor(2 ** (nbit - 1) - 1).to(x.device)
                    q = sym_quant_dequant(x, scale1, torch.tensor(2 ** (nbit - 1) - 1).to(x.device))

                    q -= x
                    q.abs_()
                    q.pow_(2.4)
                    err = torch.sum(q, 1).unsqueeze(0).unsqueeze(0)
                    tmp = err < best
                    if torch.any(tmp):
                        tmp = tmp.squeeze(0)
                        best[tmp] = (err.squeeze(0))[tmp]
                        xmax[tmp] = xmax1[tmp]
            act_dict[name]["input"] = xmax.to(torch.float16)
        else:
            tmp = torch.zeros(x.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0)
            xmin = torch.minimum(x.amin(-2, keepdim=True), tmp)
            xmax = torch.maximum(x.amax(-2, keepdim=True), tmp)
            xmax = (torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5))
            mse = True
            if mse:
                best = torch.full([x.shape[-1]], float('inf'), device=x.device).unsqueeze(0).unsqueeze(0)
                for i in range(int(0.8 * 100)):
                    p = 1 - i / 100
                    xmax1 = p * xmax
                    scale1 = xmax1 / torch.tensor(2 ** (nbit - 1) - 1).to(x.device)
                    q = sym_quant_dequant(x, scale1, torch.tensor(2 ** (nbit - 1) - 1).to(x.device))

                    q -= x
                    q.abs_()
                    q.pow_(2.4)
                    err = torch.sum(q, 1).unsqueeze(0).unsqueeze(0)
                    tmp = err < best
                    if torch.any(tmp):
                        tmp = tmp.squeeze(0)
                        best[tmp] = (err.squeeze(0))[tmp]
                        xmax[tmp] = xmax1[tmp]
            act_dict[name]["input"] =  torch.max(act_dict[name]["input"], xmax.to(torch.float16))

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "fc2" in name or "out_proj" in name:
                hooks.append(m.register_forward_hook(
                    partial(stat_io_hook_token, name=name)))
            else:
                hooks.append(m.register_forward_hook(
                    partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    # pbar = tqdm(range(num_samples))
    # dataset = load_dataset('json', data_files=dataset_path, split="train")
    # dataset = dataset.shuffle(seed=42)
    # for i in pbar:
    #     input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
    #                           max_length=seq_len, truncation=True).input_ids.to(device)
    #     model(input_ids)

    dataloader = torch.load("/home/wangjinguang/mix-opt.cache")
    i=0
    for batch in tqdm(dataloader):
        i = i + 1
        if i < 32:
            model(batch[0].to(device))
        else:
            pass

    for hook in hooks:
        hook.remove()

    return act_dict
