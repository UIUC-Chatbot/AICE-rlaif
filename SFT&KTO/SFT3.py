import torch
import transformers
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.commands.cli_utils import TrlParser
from transformers import TrainingArguments
import os
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator, DistributedType, PartialState
from dataclasses import dataclass, field
import pandas as pd

### Try gpu ###
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

parser = TrlParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_and_config()
# set use reentrant to False
if training_args.gradient_checkpointing:
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

### Accelerator ###
# accelerator = Accelerator()
# print("Total devices:", len(accelerator.state.device_ids))
# device = accelerator.device

### Quantization ###
# torch_dtype = torch.bfloat16
# quant_storage_dtype = torch.bfloat16
#
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_quant_storage=quant_storage_dtype,
# )

### Base Model ###
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
def template_dataset(examples):
    return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory = { 0: '50.0GiB', 1: '50.0GiB', 2: '50.0GiB', 3: '50.0GiB'})
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=quantization_config,
    attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
    torch_dtype=torch.float32,
    use_cache=False,  # this is needed for gradient checkpointing
    # device_map={'':torch.cuda.current_device()},
    max_memory={0: '60.0GiB', 1: '60.0GiB', 2: '60.0GiB', 3: '60.1GiB'}
)
model.gradient_checkpointing_enable()           # enable gradient checkpointing

### create PEFT-Lora model ###
# peft_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     inference_mode=False,
#     bias=None,
#     r=16,
#     lora_alpha=8,
#     lora_dropout=0.05,
#     target_modules=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj', 'down_proj'],
# )
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
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
# train_set = load_dataset("json", data_files="ft_dataset_uiuc.json", split="train")

## Build Training Samples

# def trainable_dataset(dataset):
#     output_texts = []
#     for i in range(len(dataset['prompt'])):
#         text = "Question:\n" + dataset["prompt"][i] + "\nAnswer:" + dataset["completion"][i]
#         output_texts.append(text)
#     return output_texts

# ## Use Llama Template (See AICE.ipynb for details)
#
# # Convert dataset to OAI messages
# system_message = """You are Llama, an AI assistant to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
#
#
# def create_conversation(sample):
#     sample["answer"] = [{"role": "system", "content": system_message}] + [{"role": "user", "content": sample["question"]}] + [{"role": "assistant", "content": sample["answer"]}]
#     return sample
#
# # Load dataset from the hub
# dataset = load_dataset("CAII-NCSA/ECE_RAG_critique_judge", token="")["train"]
#
# # Add system message to each conversation
# columns_to_remove = list(dataset.features)
# columns_to_remove.remove("question")
# columns_to_remove.remove("answer")
# dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)
#
# # Filter out conversations which are corrupted with wrong turns, keep which have even number of turns after adding system message
#
# # save datasets to disk
# dataset.to_json("sft_train_dataset.json", orient="records", force_ascii=False)

train_set = load_dataset("json", data_files="sft_train_dataset.json", split="train")
train_set = train_set.map(template_dataset, remove_columns=["messages"])

### Training arguments ###

# training_arguments = TrainingArguments(
#     output_dir="./finetune_log",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=2,
#     auto_find_batch_size=False,
#     # fp16=True,
#     bf16=True,
#     tf32=True,
#     gradient_checkpointing=True,
# )
if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

### Trainer ###
# device_string = PartialState().process_index

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_set,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=8192,
    tokenizer=tokenizer,
    # packing=True,
    dataset_kwargs={
        "add_special_tokens": False,    # We template with special tokens
        "append_concat_token": False,   # No need to add additional separator token
    },
)
# print(trainer.accelerator.device)

## prepare accelerator ###
# trainer = accelerator.prepare(trainer)

### Train ###
if trainer.accelerator.is_main_process:
    print("training args:", training_args)
    print("train_set:", train_set)
    print("Example:", train_set[0]["text"])
print("\nStart Training!")
trainer.train()

### Save model ###
# accelerator.wait_for_everyone()
# trainer.model.save_pretrained("Mistral_8x7B_SFT_Lora")
# trainer.tokenizer.save_pretrained("Mistral_8x7B_SFT_Lora")
# trainer.model.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=False)
# trainer.tokenizer.push_to_hub("LongQ/Llama3_UIUC_SFT_Lora", private=False)

# accelerator.wait_for_everyone()

trainer.is_fsdp_enabled = True

# trainer.accelerator.wait_for_everyone()
trainer.save_model()

try:
    print("Pushing Models to Hub...")
    trainer.model.push_to_hub("LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora_NoQuant", private=True, token="")
    trainer.tokenizer.push_to_hub("LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora_NoQuant", private=True, token="")
except:
    print("Exit Normally.")

# trainer.accelerator.wait_for_everyone()
# if trainer.accelerator.is_main_process:
#     unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
#     unwrapped_tokenizer = trainer.accelerator.unwrap_model(trainer.tokenizer)
#     if trainer.is_fsdp_enabled:
#         trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
#     unwrapped_model.push_to_hub("LongQ/Llama3_8B_UIUC_ECE_SFT_Lora", private=True, token="")
#     unwrapped_tokenizer.push_to_hub("LongQ/Llama3_8B_UIUC_ECE_SFT_Lora", private=True, token="")

### Run with Accelerator ###
## accelerate launch --config_file "./fsdp_config.yaml" SFT.py
## accelerate launch --config_file "./config.yaml" SFT.py

print("Exit Normally.")