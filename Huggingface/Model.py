from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers import Trainer, TrainingArguments
import torch
from prepare_data import prepare_REMI
import glob

def get_Model(name):
    if name=="GPT2":
        config=GPT2Config()
        model=GPT2LMHeadModel(config)
    elif name=="TransfoXLConfig":
        config=TransfoXLConfig()
        model=TransfoXLLMHeadModel(config)
    return model

model = get_Model("GPT2").to("mps")
dataset = prepare_REMI(glob.glob('Pop1K7/midi_analyzed/src_*/*.mid'))

training_args = TrainingArguments(
    output_dir="../result/GPT2_REMI",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=3,
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