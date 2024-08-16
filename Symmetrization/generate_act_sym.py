import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,AutoModelForSeq2SeqLM)
import argparse

from sym_calibration import get_static_decoder_layer_symmetrizations

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}", model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}", **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    model_name = 'opt-iml-30b'
    parser.add_argument('--model-name', type=str,
                        default=model_name, help='model name')
    parser.add_argument('--output-path', type=str, default=f'/home/wangjinguang/linuxPJ/New/symmetrizations/{model_name}.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='/home/wangjinguang/linuxPJ/smoothquant-main/dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    act_scales = get_static_decoder_layer_symmetrizations(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    # print(act_scales[f"model.decoder.layers.{0}.self_attn.q_proj"]['input'].shape)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)

    # import matplotlib.pyplot as plt
    # tensor_to_visualize = act_scales[f"model.decoder.layers.{0}.self_attn.q_proj"]['input']
    # tensor_data = tensor_to_visualize.cpu().detach().numpy().flatten()
    # plt.hist(tensor_data, bins=50, alpha=0.7, color='blue')
    # plt.title('Histogram of attn_input_scale')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()


if __name__ == '__main__':
    main()
