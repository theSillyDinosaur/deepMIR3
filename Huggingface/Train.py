from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
from prepare_data import prepare_REMI
import glob
import argparse
import os


def get_Model(name):
    if name=="GPT2":
        config=GPT2Config()
        model=GPT2LMHeadModel(config)
    elif name=="TransfoXL":
        config=TransfoXLConfig(
            n_head=12,
            n_layer=12
        )
        model=TransfoXLLMHeadModel(config)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["GPT2", "TransfoXL"], default='GPT2')
parser.add_argument("--pop1k7", type=str, default='Pop1K7_REMI')
parser.add_argument("--ckpt", type=str, default='result')
parser.add_argument("--n_epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

model = get_Model(args.model)
dataset = load_from_disk(args.pop1k7)

training_args = TrainingArguments(
    output_dir=args.ckpt,
    evaluation_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.n_epoch,
    warmup_steps = 16000,
    save_steps=16000
)

# Initialize Trainer without specifying the tokenizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("start training!")

trainer.train()