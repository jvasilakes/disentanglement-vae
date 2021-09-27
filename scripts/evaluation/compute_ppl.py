import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing {train,dev,test}.jsonl")
    parser.add_argument("-N", type=int, default=-1)
    return parser.parse_args()


def main(data_dir, max_n=-1):
    # {dataset_name: [sentences]}
    print("Loading data...", end='', flush=True)
    data = get_data(data_dir)
    print("done", flush=True)

    print("Loading GPT2...", end='', flush=True)
    device = "cuda"
    model_id = "gpt2-large"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    print("done", flush=True)

    ppls = defaultdict(list)
    for (dataset_name, sents) in data.items():
        i = 0
        for sent in tqdm(sents, desc=dataset_name):
            if i == max_n:
                break
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                log_likelihood = outputs.loss.detach()
                ppl = torch.exp(log_likelihood)
            ppls[dataset_name].append(ppl)
            i += 1

    for (dataset_name, ppl_list) in ppls.items():
        avg_ppl = torch.stack(ppl_list).mean()
        print(f"{dataset_name}: {avg_ppl:.4f}")


def get_data(data_dir):
    dataset_names = ["train", "dev", "test"]
    output = {}
    for name in dataset_names:
        fname = os.path.join(data_dir, f"{name}.jsonl")
        data = [json.loads(line) for line in open(fname)]
        sentences = [datum["sentence"] for datum in data]
        output[name] = sentences
    return output


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, max_n=args.N)
