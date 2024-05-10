import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import os
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator, DistributedType, PartialState
import pandas as pd

### Try gpu ###
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

### Accelerator ###
# accelerator = Accelerator()
# print("Total devices:", len(accelerator.state.device_ids))
# device = accelerator.device

### Quantization ###
torch_dtype = torch.bfloat16
quant_storage_dtype = torch.bfloat16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=quant_storage_dtype,
)

### Base Model ###
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory = { 0: '50.0GiB', 1: '50.0GiB', 2: '50.0GiB', 3: '50.0GiB'})
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = quantization_config,
    attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
    torch_dtype=quant_storage_dtype,
    use_cache=False,  # this is needed for gradient checkpointing
    device_map="auto"
)
model.gradient_checkpointing_enable()           # enable gradient checkpointing

### create PEFT-Lora model ###
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj', 'down_proj'],
)
# model = get_peft_model(model, peft_config)
# print("PEFT-Lora Model created:")
# model.print_trainable_parameters()

### Build Dataset : See Jupyter Notebook & ft_dataset.json ###

# ## TruthfulQA
# # Use 300 samples from TruthfulQA for training
# data_start = 100
# data_end = 400
# train_questions = load_dataset("truthful_qa", "generation")["validation"][data_start:data_end]["question"]
# train_answers = load_dataset("truthful_qa", "generation")["validation"][data_start:data_end]["best_answer"]
#
# # with open("ft_dataset.json", "w", encoding='utf-8') as f:
# #     for idx in range(len(train_questions)):
# #         f.write(json.dumps({"prompt": train_questions[idx], "completion": train_answers[idx]}))
# #         f.write("\n")

# train_set = load_dataset("json", data_files="ft_dataset.json", split="train")
# train_set = load_dataset("LongQ/TruthfulQA_critique_small", split="train")

## UIUC.CHAT
# data_file = "./Judge_result.csv"
# data_csv = pd.read_csv(data_file)
#
# questions = data_csv['question'].to_list()
# answers = data_csv['answer'].to_list()
# scores = data_csv['llm_judge_score'].to_list()
#
# # Data Cleaning
# idx = 0
# while idx < len(questions):
#     if type(questions[idx]) != str or type(answers[idx]) != str or type(scores[idx]) != float or scores[idx] > 10 or scores[idx] < 0 or scores[idx] != scores[idx]:
#         del questions[idx]
#         del answers[idx]
#         del scores[idx]
#         continue
#     else:
#         idx += 1
#
# print("Load dataset of length :", len(questions))
#
# # Select only score >= 8
# questions_good = []
# answers_good = []
# for idx in range(len(questions)):
#     if scores[idx] >= 8:
#         questions_good.append(questions[idx])
#         answers_good.append(answers[idx])
#
# print("Filter good dataset of length :", len(questions_good))

# # with open("ft_dataset.json", "w", encoding='utf-8') as f:
# #     for idx in range(len(train_questions)):
# #         f.write(json.dumps({"prompt": train_questions[idx], "completion": train_answers[idx]}))
# #         f.write("\n")

# See AICE.ipynb for details
train_set = load_dataset("json", data_files="ft_dataset_uiuc.json", split="train")

## Build Training Samples

def trainable_dataset(dataset):
    output_texts = []
    for i in range(len(dataset['prompt'])):
        text = "Question:\n" + dataset["prompt"][i] + "\nAnswer:" + dataset["completion"][i]
        output_texts.append(text)
    return output_texts

### Training arguments ###

training_arguments = TrainingArguments(
    output_dir="./finetune_log",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    auto_find_batch_size=False,
    # fp16=True,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
)
if training_arguments.gradient_checkpointing:
    model.gradient_checkpointing_enable()

### Trainer ###
# device_string = PartialState().process_index

trainer = SFTTrainer(
    model,
    train_dataset=train_set,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=trainable_dataset,
    max_seq_length=8192,
    packing=True,
    peft_config=peft_config,
)
print(trainer.accelerator.device)

## prepare accelerator ###
# trainer = accelerator.prepare(trainer)
# if trainer.is_fsdp_enabled:
#     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

### Train ###
print("\nStart Training!")
trainer.is_fsdp_enabled = True
trainer.train()

### Save model ###
# accelerator.wait_for_everyone()
# trainer.model.save_pretrained("Mistral_8x7B_SFT_Lora")
# trainer.tokenizer.save_pretrained("Mistral_8x7B_SFT_Lora")
# trainer.model.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=False)
# trainer.tokenizer.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=False)

# accelerator.wait_for_everyone()

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
trainer.model.push_to_hub("LongQ/Llama3_70B_UIUC_SFT_Lora", private=False)
trainer.tokenizer.push_to_hub("LongQ/Llama3_70B_UIUC_SFT_Lora", private=False)


# if accelerator.is_main_process:
#     unwrapped_model = accelerator.unwrap_model(trainer.model)
#     unwrapped_tokenizer = accelerator.unwrap_model(trainer.tokenizer)
#     if trainer.is_fsdp_enabled:
#         trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
#     unwrapped_model.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=True, token="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")
#     unwrapped_tokenizer.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=True, token="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")

### Run with Accelerator ###
## accelerate launch --config_file "./fsdp_config.yaml" SFT.py
## accelerate launch --config_file "./config.yaml" SFT.py
