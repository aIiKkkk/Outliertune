import torch
from torch import nn
from typing import Optional, Tuple, List
from smoothquant.fake_quant import W8A8Linear

class CalScaleOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        embed_dim: int,
        num_heads: int,
        weight_quant,
        act_quant,
        quantize_bmm_input,
        i,
        act_scales,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = 0.0
        self.head_dim = embed_dim // num_heads
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.quantize_bmm_input = quantize_bmm_input

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = True

        self.k_proj = W8A8Linear.from_float(org_module.k_proj, weight_quant=weight_quant, act_quant=act_quant)
        self.v_proj = W8A8Linear.from_float(org_module.v_proj, weight_quant=weight_quant,
                                            act_quant=act_quant, quantize_output=quantize_bmm_input)
        self.q_proj = W8A8Linear.from_float(org_module.q_proj, weight_quant=weight_quant, act_quant=act_quant)
        # qk的输出没有进行量化，后续对qk进行缩放之后再进行量化
        self.out_proj = W8A8Linear.from_float(org_module.out_proj, weight_quant=weight_quant, act_quant=act_quant)
        ## 先不考虑 把QK暂时不进行量化，放到后面量化

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # print(key_states.shape)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # 计算缩放后的注意力头的计算过程，注意计算掩码之后的误差







            mixed = True
            if mixed:
                proj_shape = (bsz * self.num_heads, -1, self.head_dim)
                query_states1 = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
                key_states1 = self._shape(key_states, -1, bsz).view(*proj_shape)
                src_len = key_states1.size(1)
                attn_weights = torch.bmm(query_states1, key_states1.transpose(1, 2))
                if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                        f" {attn_weights.size()}")
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
                    attn_weights = (attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask)
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
                    attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

                heavy_budget = int(attn_weights.shape[-1])
                row_vector = torch.tensor([heavy_budget - i if i < heavy_budget else 1 for i in range(heavy_budget)],
                                          dtype=torch.float16)
                row_vector = torch.unsqueeze(row_vector, dim=0).to(attn_weights.device)

                if attn_weights.dtype == torch.float16:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                else:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1)

                accumulated_attention_score = torch.sum(tmp_attn[:, :, :], dim=-2)
                accumulated_attention_score = torch.unsqueeze(torch.sum(accumulated_attention_score, dim=0), dim=0)
                accumulated_attention_score = accumulated_attention_score / (row_vector)
                accumulated_attention_score = accumulated_attention_score / (bsz * self.num_heads)

                Group = True
                if Group:
                    ratios = [0.25, 0.25, 0.25, 0.25]
                    num_ranges = len(ratios)
                    ranges = []
                    start = 1
                    for i, ratio in enumerate(ratios):
                        range_size = int(heavy_budget * ratio)
                        end = min(start + range_size - 1, heavy_budget) if i < num_ranges - 1 else heavy_budget
                        ranges.append((start, end))
                        start = end + 1
                    topk_indices_list = []
                    for start, end in ranges:
                        partial_score = accumulated_attention_score[:, start - 1:end]
                        _, topk_indices = partial_score.topk(k=int(heavy_budget / 40), dim=-1)
                        topk_indices_list.append(topk_indices + start - 1)
                    tmp_topk_index = torch.cat(topk_indices_list, dim=-1)
                else:
                    _, tmp_topk_index = accumulated_attention_score.topk(k=int(heavy_budget / 1.2), dim=-1)
                    # tmp_topk_index = torch.randint(low=0, high=heavy_budget, size=(1, int(heavy_budget / 2))).to(attn_weights.device) # 随机选取

                def fake_quant_mix(t, axis, n_bits=8):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s

                def fake_quant(t, axis, n_bits=4):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max(dim=axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t
                def fake_quant_v(t, axis, n_bits=8):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max(dim=axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t

                key_states_importance = fake_quant_mix(key_states, axis=-1)
                key_states = fake_quant(key_states, axis=-1)
                key_states[:, tmp_topk_index, :] = key_states_importance[:, tmp_topk_index, :]
                value_states = fake_quant_v(value_states, axis=-2)
                #print(tmp_topk_index.shape)
                #print(key_states.shape)
                #print(key_states_importance.shape)
            else:
                def fake_quant2(t, axis, n_bits=3):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max(dim=axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t
                key_states = fake_quant2(key_states, axis = -1)
                value_states = fake_quant2(value_states, axis = -2)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)


        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value