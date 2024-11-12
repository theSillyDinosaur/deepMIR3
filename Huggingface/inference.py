from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
from prepare_data import prepare_REMI
import glob
import argparse
import os
import miditok
from tqdm import *


def get_Model(name, ckpt):
    if name=="GPT2":
        config=GPT2Config(
        )
        model=GPT2LMHeadModel(config).from_pretrained(ckpt)
    elif name=="TransfoXL":
        config=TransfoXLConfig(
            n_head=12,
            n_layer=12
        )
        model=TransfoXLLMHeadModel(config).from_pretrained(ckpt)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["GPT2", "TransfoXL"], default='GPT2')
parser.add_argument("--ckpt", type=str, default='ckpt')
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

special_start = 0
pitch_start = 5
velocity_start = 93
duration_start = 125
position_start = 189
pitchDrum_start=221

special_mask = torch.asarray(5*[True]+(282-5)*[False])
pitch_mask = torch.asarray(5*[False]+(93-5)*[True]+(282-93)*[False])
velocity_mask = torch.asarray(93*[False]+(125-93)*[True]+(282-125)*[False])
duration_mask = torch.asarray(125*[False]+(189-125)*[True]+(282-189)*[False])
position_mask = torch.asarray(189*[False]+(221-189)*[True]+(282-221)*[False])
pitchDrum_mask = torch.asarray(221*[False]+(282-221)*[True])

top_p = 0.9
top_k = 50
Temp = 1

model = get_Model(args.model, args.ckpt)
tokenizer = miditok.REMI()
# print(tokenizer.vocab)

tokens = torch.asarray([  1,   4,   4, 189,  31, 106, 141,  50, 108, 141,  38, 104, 140,  41,
        104, 141,  43, 104, 132,  46, 104, 141, 205,  33, 106, 140,  48, 108,
        160,  52, 108, 142,  40, 105, 141,  43, 104, 155,  45, 105, 140,   4,
        189,  33, 104, 140,  41, 104, 140,  45, 104, 140,  50, 108, 140, 205,
         33, 107, 130,  41, 106, 130,  45, 108, 130,  50, 108, 138, 211,  33,
        109, 130,  52, 111, 130,  26, 107, 129,  41, 108, 129,  45, 107, 129,
         48, 106, 134, 217,  26, 107, 129,  41, 107, 144,  45, 108, 129,  52,
        110, 129,  33, 109, 128,   4, 189,  31, 107, 125,  43, 108, 140,  46,
        109, 141,  50, 109, 141,  38, 107, 136, 205,  48, 105, 159,  52, 107,
        143,  33])

# tokens = torch.asarray([1, 4])
bar_count = 1
event_count = 2

pbar = tqdm()
pbar.set_description(f"event = {event_count}, bar = {bar_count}")
note_cnt = 0

while bar_count < 32:
    logit = model(tokens[-1024:].unsqueeze(0))[0][-1][-1][:282]
    if tokens[-1] < 5:
        mask = special_mask + pitch_mask + pitchDrum_mask + position_mask
        note_cnt = 0
    elif tokens[-1] < 93:
        mask = velocity_mask
        note_cnt += 1
    elif tokens[-1] < 125:
        mask = duration_mask
    elif tokens[-1] < 189:
        mask = special_mask + pitch_mask + pitchDrum_mask + position_mask
    elif tokens[-1] < 221:
        mask = pitch_mask + pitchDrum_mask
        note_cnt = 0
    else:
        mask = velocity_mask
        note_cnt += 1

    probabilities = torch.softmax(logit/Temp, dim=-1) * mask
    probabilities = probabilities + 2*(2**note_cnt) * probabilities * (special_mask + position_mask)
    probabilities = probabilities / torch.sum(probabilities)

    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
    else:
        top_k_probs, top_k_indices = probabilities, torch.arange(len(probabilities))
    top_k_probs = top_k_probs / torch.sum(top_k_probs)
    sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
    sorted_probs = sorted_probs / torch.sum(sorted_probs)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False)

    sorted_probs = sorted_probs[:cutoff_index + 1]
    sorted_probs = sorted_probs/torch.sum(sorted_probs)
    sorted_indices = top_k_indices[sorted_indices[:cutoff_index + 1]]
    if sorted_probs.shape[0] == 0:
        print(probabilities, top_k_probs, cumulative_probs, sorted_indices)

    next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
    if next_token == 4:
        bar_count += 1
    tokens = torch.cat((tokens, next_token), dim=-1)
    event_count += 1
    pbar.set_description(f"event = {event_count}, bar = {bar_count}, note_cnt = {note_cnt}")
    pbar.update()
print(tokens)
score = tokenizer.decode([tokens.tolist()])
score.dump_midi("test.mid")
