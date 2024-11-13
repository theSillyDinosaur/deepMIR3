Python version: 3.9.20

The inference code is:
python3 Huggingface/inference.py --ckpt [ckpt] --model [model] --prompt [prompt directory] --n_bar [int] --sample [int] --output [result directory]

* The option of ckpt is [gpt2_ckpt, transfoXL_ckpt, transfoL_ckpt], and the option of the model is [GPT2, TransfoXL, TransfoL].
* If the prompt is empty, the default prompt will be ["start", "bar"]. Otherwise the code will go through the midi file in the directory.
* adjust sample=p to generate p results for every prompt
* The file is stored at [result directory] in output.

For example, to run the task 2 with directory "prompt_song":
python3 Huggingface/inference.py --ckpt gpt2_ckpt --model GPT2 --prompt prompt_song --n_bar 24 --sample 1

The produced result is in result directory, with "{model}_free" be default prompt and "GPT2_song" be song continuation