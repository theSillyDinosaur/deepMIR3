from datasets import load_dataset
import glob
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI
from datasets import Dataset
from tqdm import *
import torch
import json


def prepare_REMI(data_list, need_attnMask=False):
    tokenizer = REMI()

    dataset = DatasetMIDI(
        files_paths=data_list,
        tokenizer=tokenizer,
        max_seq_len=32768,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )

    data = {"input_ids": [], "labels": []}
    if need_attnMask:
        data["attention_mask"] = []
    max_bar = 0
    pbar = tqdm(dataset)
    for song in pbar:  # iterate through each tokenized song
        tokens = song["input_ids"]
        bar_index = torch.cat((torch.asarray([0]), (tokens == 4).nonzero().squeeze(1)), dim=0)
        index_i = 0
        while(bar_index[index_i]+1024 < tokens.shape[0]):
            data["input_ids"].append(tokens[bar_index[index_i]:bar_index[index_i]+1024])
            data["labels"].append(tokens[bar_index[index_i]+1:bar_index[index_i]+1025])
            if need_attnMask:
                data["attention_mask"].append(torch.asarray(1024*[1]))
            index_i += 1

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data)
    split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    return split_dataset

if __name__ == "__main__":
    prepare_REMI(glob.glob('Pop1K7/midi_analyzed/src_*/*.mid'), need_attnMask=True)