import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from Smoothquant.smooth import smooth_lm
from GPTQ import opt_quantize
# from lm_evaluation.lm_eval import tasks, evaluator
from DecoderLayer import QuantOPTDecoderLayer
from tqdm import tqdm
import torch.nn as nn
import os
from pprint import pprint
from opt import OPTClass
from symmetrization import symmetrization_lm
from datasets import load_from_disk
import argparse

activation_para = 8

def quantize_model(model, act_scales, weight_quant='per_channel', act_quant='per_channel', quantize_bmm_input=True):
    embed_dim = model.model.decoder.layers[0].embed_dim
    num_heads = model.model.decoder.layers[0].self_attn.num_heads
    for i in tqdm(range(len(model.model.decoder.layers))):
        model.model.decoder.layers[i] = QuantOPTDecoderLayer(model.model.decoder.layers[i],
                             embed_dim, num_heads, weight_quant, act_quant, quantize_bmm_input, model.model.config, act_scales, i, activation_para)
    torch.cuda.empty_cache()
    return model

def parser_gen():
    parser = argparse.ArgumentParser()
    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=activation_para,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', default=True,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', default=True,
                        help='''Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', default=True,
                        help='act-order in GPTQ')
    parser.add_argument('--lm_eval_batch_size', type=int, default=16,
                        help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa","hellaswag", "arc_easy", "arc_challenge", "winogrande","boolq","copa","rte"],
    )
    #
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, tokenizer, device):
        # self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # def tokenize_function(examples):
        #     example = self.tokenizer(examples['text'])
        #     return example

        # self.dataset = self.dataset.map(tokenize_function, batched=True)
        # self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        for dataset in ["wikitext2", "ptb", "c4"]:

            cache_testloader = f"/home/wangjinguang/linuxPJ/smoothquant-main/{dataset}_testloader_opt_all.cache"

            testloader = torch.load(cache_testloader)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            seqlen = 2048
            nsamples = testenc.numel() // seqlen
            model.config.use_cache = False
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
                outputs = model.model.decoder(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                               :, 1:
                               ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            print(dataset, ppl.item())
        return ppl

model_name = "opt-66b"

tokenizer = GPT2Tokenizer.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}")
#dataset = load_from_disk("/data/wangjinguang/dataset/lambada_openai")
#print("Loaded dataset from {/data/wangjinguang/dataset/lambada_openai}")
evaluator_PPL = Evaluator(tokenizer, 'cuda')
# model = OPTForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}",torch_dtype=torch.float16, device_map='auto')
# acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model)
# lm = OPTClass(model, tokenizer)
# t_results = evaluator.simple_evaluate(lm, tasks="lambada_standard,winogrande,rte,piqa,openbookqa,boolq,hellaswag,arc_easy,arc_challenge,copa",
#                                             num_fewshot=0, limit=None)
# pprint(t_results)
args = parser_gen()


compare_LLMint8 = False
if compare_LLMint8:
    model_8bit = AutoModelForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}",
                                                      device_map="auto", load_in_8bit=True)

    print("Starting evaluate")
    acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_8bit)
    # print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')
    # lm = OPTClass(model_8bit, tokenizer)
    # t_results = evaluator.simple_evaluate(lm, tasks="lambada_standard",
    #                                         num_fewshot=0, limit=None)
    # pprint(t_results)
# ,winogrande,rte,piqa,openbookqa,boolq,hellaswag,arc_easy,arc_challenge,copa

compare_smoothw8a8 = True
if compare_smoothw8a8:
    model = OPTForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}",
                                        torch_dtype=torch.float16, device_map='auto')

    # print(model)
    print(model.model.decoder.layers[0].self_attn.k_proj.weight.dtype)

    # 导入对称化需要的迁移因子
    scales = torch.load(f'/home/wangjinguang/linuxPJ/New/symmetrizations/{model_name}.pt')
    # scales = torch.load("/home/wangjinguang/linuxPJ/Outlier_Suppression_Plus-main/exp/opt/scale_shift_list.pth")
    # scales = scales["shift_list"]
    symmetrization_lm(model, scales)

    ## 导入对称化之后激活的缩放因子。在消融实验中先不进行对称化，直接获得原始激活的缩放因子。
    act_scales = torch.load(f'/home/wangjinguang/linuxPJ/New/act_scales_sym/{model_name}-int8.pt')
    ## 权重更新操作
    smooth_lm(model, act_scales, activation_para)

    ### 更新完成后，对权重进行量化，并使用已存储的激活缩放因子对激活进行静态量化。
    print("Starting quantize_activations")
    model_smoothquant_w8a8 = quantize_model(model, act_scales)

    dataloader = torch.load("/home/wangjinguang/mix-opt.cache")
    print("Starting GPTQ Quantizing ...")
    quantizers = opt_quantize(model_smoothquant_w8a8, dataloader, args)


    print("Starting evaluate")
    acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_smoothquant_w8a8)

    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(pretrained=model_smoothquant_w8a8, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
    results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, batch_size=args.lm_eval_batch_size)['results']

    pprint(results)
    # print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')
    # lm = OPTClass(model_smoothquant_w8a8, tokenizer)
    # t_results = evaluator.simple_evaluate(lm, tasks="lambada_standard,winogrande,rte,piqa,openbookqa,boolq,hellaswag,arc_easy,arc_challenge,copa",
    #                                         num_fewshot=0, limit=None)
    # pprint(t_results)



