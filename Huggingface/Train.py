from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers import Trainer, TrainingArguments
import torch
from prepare_data import prepare_REMI
import glob
import argparse
import os


def get_Model(name):
    if name=="GPT2":
        config=GPT2Config()
        model=GPT2LMHeadModel(config)
    elif name=="TransfoXL":
        config=TransfoXLConfig()
        model=TransfoXLLMHeadModel(config)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["GPT2, TransfoXL"], default='GPT2')
parser.add_argument("--pop1k7", type=str, default='Pop1K7')
parser.add_argument("--ckpt", type=str, default='result')
parser.add_argument("--n_epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

model = get_Model(args.model)
dataset = prepare_REMI(glob.glob(os.path.join(args.pop1k7, 'midi_analyzed/src_*/*.mid')))

training_args = TrainingArguments(
    output_dir=args.ckpt,
    evaluation_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.n_epoch,
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