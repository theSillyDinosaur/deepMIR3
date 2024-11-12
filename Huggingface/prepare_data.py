from datasets import load_from_disk
import glob
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI
from datasets import Dataset
from tqdm import *
import torch
import json


def prepare_REMI(data_list, need_attnMask=False, from_scratch=False, upload=False):
    if(from_scratch):
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
        pbar = tqdm(dataset)
        for song in pbar:  # iterate through each tokenized song
            tokens = song["input_ids"]
            index_i = 0
            while(index_i+1024 < tokens.shape[0]):
                data["input_ids"].append(tokens[index_i:index_i+1024])
                data["labels"].append(tokens[index_i+1:index_i+1025])
                if need_attnMask:
                    data["attention_mask"].append(torch.asarray(1024*[1]))
                index_i += 512
            print(tokens[0:128])

        # Convert to Hugging Face Dataset
        hf_dataset = Dataset.from_dict(data)
        split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
        split_dataset.save_to_disk("Pop1K7_REMI")
    
    

    return split_dataset

if __name__ == "__main__":
    prepare_REMI(glob.glob('Pop1K7/midi_analyzed/src_*/*.mid'), need_attnMask=True, from_scratch=True, upload=True)

    split_dataset = load_from_disk("Pop1K7_REMI")
    print(split_dataset)