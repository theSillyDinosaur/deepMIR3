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
import symusic as sm
from eval_metrics import prepare_data, compute_piece_pitch_entropy, compute_piece_groove_similarity


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
    elif name=="TransfoL":
        config=TransfoXLConfig(
            n_head=6,
            n_layer=6
        )
        model=TransfoXLLMHeadModel(config).from_pretrained(ckpt)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["GPT2", "TransfoXL", "TransfoL"], default='GPT2')
parser.add_argument("--ckpt", type=str, default='ckpt')
parser.add_argument("--resume", action="store_true")
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--temp", type=float, default=1)
parser.add_argument("--sample", type=int, default=20)
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--output", type=str, default="result")
parser.add_argument("--n_bar", type=int, default=32)
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

model = get_Model(args.model, args.ckpt)
tokenizer = miditok.REMI()
# print(tokenizer.vocab)

if args.prompt != None:
    if os.path.isdir(args.prompt):
        midi_set = list(filter(lambda a: a[-3:] == "mid" or a[-4:] == "midi", os.listdir(args.prompt)))
        tokens_set = []
        for midi_dir in midi_set:
            midi = sm.Score(os.path.join(args.prompt, midi_dir))
            tokens_set.append(torch.asarray(tokenizer.midi_to_tokens(midi)[0]))
        print(f"Now run on {midi_set} in {args.prompt}")
    else:
        assert args.prompt[-3:] == "mid" or args.prompt[-4:] == "midi"
        midi = sm.Score(args.prompt)
        tokens_set = [torch.asarray(tokenizer.midi_to_tokens(midi)[0])]
        print(f"Now run on {args.prompt}")
else:
    tokens_set = [torch.asarray([1, 4])]
    print("Now reset")

for prompt_idx, tokens in enumerate(tokens_set):
    if os.path.isdir(os.path.join(args.output, str(prompt_idx))) == False:
        os.makedirs(os.path.join(args.output, str(prompt_idx)))
    h4_total = 0
    gs_total = 0
    for sample_idx in range(args.sample):
        bar_count = 0
        event_count = 2

        pbar = tqdm()
        pbar.set_description(f"sample = {prompt_idx}-{sample_idx}, event = {event_count}, bar = {bar_count}/{args.n_bar}")
        while bar_count < args.n_bar:
            logit = model(tokens[-1024:].unsqueeze(0))[0][-1][-1][:282]
            if tokens[-1] < 5:
                mask = special_mask + pitch_mask + pitchDrum_mask + position_mask
            elif tokens[-1] < 93:
                mask = velocity_mask
            elif tokens[-1] < 125:
                mask = duration_mask
            elif tokens[-1] < 189:
                mask = special_mask + pitch_mask + pitchDrum_mask + position_mask
            elif tokens[-1] < 221:
                mask = pitch_mask + pitchDrum_mask
            else:
                mask = velocity_mask

            probabilities = torch.softmax(logit/args.temp, dim=-1) * mask
            probabilities = probabilities / torch.sum(probabilities)

            if args.top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probabilities, args.top_k)
            else:
                top_k_probs, top_k_indices = probabilities, torch.arange(len(probabilities))
            top_k_probs = top_k_probs / torch.sum(top_k_probs)
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
            sorted_probs = sorted_probs / torch.sum(sorted_probs)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff_index = torch.searchsorted(cumulative_probs, args.top_p, right=False)

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
            pbar.set_description(f"sample = {prompt_idx}-{sample_idx}, event = {event_count}, bar = {bar_count}/{args.n_bar}")
            pbar.update()
        score = tokenizer.decode([tokens.tolist()])

        p = os.path.join(args.output, str(prompt_idx), (f"{sample_idx}.mid"))
        score.dump_midi(os.path.join(args.output, str(prompt_idx), (f"{sample_idx}.mid")))

        seq = prepare_data(p)

        h4 = compute_piece_pitch_entropy(seq, 4)
        gs = compute_piece_groove_similarity(seq)
        print(f"Result of {prompt_idx}-{sample_idx}: h4={h4}, gs={gs}")
        h4_total += h4
        gs_total += gs
    print(f"Result of prompt {prompt_idx}: h4={h4_total/args.sample}, gs={gs_total/args.sample}")
