import torch
import os
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer
)
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
import argparse
from smooth_bloom import smooth_bloomlm
from act_calibration import get_static_decoder_layer_scales
from symmetrization import symmetrization_lm

model_name = 'opt-iml-30b'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str,
                        default=f'/home/wangjinguang/linuxPJ/New/act_scales_sym/{model_name}.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='/home/wangjinguang/linuxPJ/smoothquant-main/dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model = OPTForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}",
                                        torch_dtype=torch.float16, device_map='auto')
    tokenizer = GPT2Tokenizer.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}")

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    scales = torch.load(f'/home/wangjinguang/linuxPJ/New/symmetrizations/{model_name}.pt')
    # scales = torch.load("/home/wangjinguang/linuxPJ/Outlier_Suppression_Plus-main/exp/opt/scale_shift_list.pth")
    # scales = scales["shift_list"]
    symmetrization_lm(model, scales)

    act_scales = get_static_decoder_layer_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)



if __name__ == '__main__':
    main()