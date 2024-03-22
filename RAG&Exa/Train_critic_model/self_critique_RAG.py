from huggingface_hub import login
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import time

login(
token="XXXXXX", # ADD YOUR TOKEN HERE
add_to_git_credential=True
)
print("Loading dataset…………………………")
dataset = load_dataset("json", data_files="train_dataset.json", split="train[:1000]")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# # Load model and tokenizer
print("Loading model…………………………")
model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
print("Loading tokenizer…………………………")
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", use_fast=True)
#tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir="self-critique-RAG", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=3,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

max_seq_length = 3072 # max sequence length for model and packing of the dataset
print("Creating trainer…………………………")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)
print("Trainer created!")
print("Training…………………………")
tokenizer.pad_token = tokenizer.eos_token
t0 = time.time()
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
print(f"Training took {time.time()-t0} seconds")
# save model
trainer.save_model()

# del model
# del trainer
# torch.cuda.empty_cache()




