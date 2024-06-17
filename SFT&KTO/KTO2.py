from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from trl.commands.cli_utils import TrlParser
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, AutoPeftModelForCausalLM, cast_mixed_precision_params
from accelerate import Accelerator, PartialState
from dataclasses import dataclass, field
import pandas as pd
import yaml

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
        default=None, metadata={"help": "Model ID to use for KTO training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for KTO Trainer"}
    )

# parser = TrlParser(KTOConfig)
# kto_config = parser.parse_args_and_config()


parser = HfArgumentParser((ScriptArguments, KTOConfig))
args, kto_config = parser.parse_args_into_dataclasses()
if kto_config.gradient_checkpointing:
    kto_config.gradient_checkpointing_kwargs = {"use_reentrant": True}
# with open("fsdp_kto_config2.yaml", "w") as f:
#     args, kto_config = parser.parse_dict(yaml.safe_load(f))

# ### accelerator
# accelerator = Accelerator()

### Build Dataset ###

# ## TruthfulQA Dataset
# tf_dataset = load_dataset("truthful_qa", "generation")["validation"][0:100]
# train_dataset = load_dataset("LongQ/TruthfulQA_critique_small", split="train")

# prompts = []
# completions = []
# labels = []
#
# for q_idx in range(len(train_dataset["question"])):
#
#     ## store best answer
#     prompts.append(train_dataset["question"][q_idx])
#     completions.append("The answer is " + train_dataset["final_answer"][q_idx])
#     labels.append(True)
#
# for q_idx in range(len(tf_dataset["question"])):
#
#     ## store best answer
#     # prompts.append(tf_dataset["question"][q_idx])
#     # completions.append("The answer is " + tf_dataset["best_answer"][q_idx])
#     # labels.append(True)
#
#     # ## store correct answer
#     # for corr_idx in range(len(train_dataset["correct_answers"][q_idx])):
#     #     prompts.append(train_dataset["question"][q_idx])
#     #     completions.append(train_dataset["correct_answers"][q_idx][corr_idx])
#     #     labels.append(True)
#
#     ## store incorrect answer
#     for incorr_idx in range(len(tf_dataset["incorrect_answers"][q_idx])):
#         prompts.append(tf_dataset["question"][q_idx])
#         completions.append("The answer is " +tf_dataset["incorrect_answers"][q_idx][incorr_idx])
#         labels.append(False)

# ## UIUC.CHAT Dataset
# data_file = "./final_judge.csv"
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
# # Select only score>=7
# questions_best = []
# answers_best = []
# for idx in range(len(questions)):
#     if scores[idx] >= 7:
#         questions_best.append(questions[idx])
#         answers_best.append(answers[idx])
#
# print("Filter best dataset of length :", len(questions_best))
#
# ## Select only score<=3
# questions_bad = []
# answers_bad = []
# for idx in range(len(questions)):
#     if scores[idx] <= 3:
#         questions_bad.append(questions[idx])
#         answers_bad.append(answers[idx])
#
# print("Filter bad dataset of length :", len(questions_bad))
#
# ## Labeling Dataset
# prompts = []
# completions = []
# labels = []
#
# for q_idx in range(len(questions_best)):
#
#     ## store best answer
#     prompts.append("Question:\n" + questions_best[q_idx] + "\nAnswer:")
#     completions.append(answers_best[q_idx])
#     labels.append(True)
#
# for q_idx in range(len(questions_bad)):
#
#     ## store best answer
#     prompts.append("Question:\n" + questions_bad[q_idx] + "\nAnswer:")
#     completions.append(answers_bad[q_idx])
#     labels.append(False)
#
# print("Number of samples :", len(prompts))
#
# kto_dataset_dict = {"prompt":prompts, "completion":completions, "label":labels}
# train_dataset = Dataset.from_dict(kto_dataset_dict)

### Load Model ###

### Quantization ###
torch_dtype = torch.float16
quant_storage_dtype = torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=quant_storage_dtype,
)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_quant_storage=quant_storage_dtype,
# )

# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_id = "LongQ/Mistral_SFT_TFQA"

lora_model_id = "LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora"
# lora_model_id = "LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora"

merged_model_id = "LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora_Merged"

LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

tokenizer = AutoTokenizer.from_pretrained(lora_model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.add_eos_token = True
tokenizer.add_bos_token = True
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
def template_dataset(examples):
  return {"prompt": tokenizer.apply_chat_template(examples["prompt"], tokenize=False), "completion": tokenizer.apply_chat_template(examples["completion"], tokenize=False), "label":examples["label"]}

### Load Base Model ###
# base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: '30.0GiB', 1: '30.0GiB'})
# base_model = AutoModelForCausalLM.from_pretrained(model_id)
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=quantization_config,
#     attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
#     torch_dtype=quant_storage_dtype,
#     use_cache=False,  # this is needed for gradient checkpointing
#     device_map={'':torch.cuda.current_device()},
# )

### Load Merged Model ###
# model = AutoModelForCausalLM.from_pretrained(
#     merged_model_id,
#     token="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",
#     # quantization_config=quantization_config,
#     attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
#     torch_dtype=torch_dtype,
#     use_cache=False,  # this is needed for gradient checkpointing
#     # device_map={'':torch.cuda.current_device()},
# )

### Load Lora Model and Merge ###
# model = AutoPeftModelForCausalLM.from_pretrained(
#     lora_model_id,
#     token="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",
#     # quantization_config=quantization_config,
#     attn_implementation="sdpa",
#     torch_dtype=torch.float32,
#     # load_in_4bit=True,
#     is_trainable=True,
#     # use_cache=False,
#     # device_map={'':torch.cuda.current_device()},
# )

model = AutoPeftModelForCausalLM.from_pretrained(
    lora_model_id, # location of saved SFT model
    token="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",
    # low_cpu_mem_usage=True,
    # quantization_config=quantization_config,
    torch_dtype=torch.float8,
    # load_in_4bit=True,
    # is_trainable=True,
)
# model_ref = AutoPeftModelForCausalLM.from_pretrained(
#     lora_model_id, # same model as the main one
#     token="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",
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
#     load_in_4bit=True,
# )


# model_ref = AutoPeftModelForCausalLM.from_pretrained(
#     lora_model_id,
#     token="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",
#     # quantization_config=quantization_config,
#     attn_implementation="sdpa",
#     torch_dtype=torch.float16,
#     # load_in_4bit=True,
#     # is_trainable=True,
#     # use_cache=False,
#     # device_map={'':torch.cuda.current_device()},
# )

model.gradient_checkpointing_enable()           # enable gradient checkpointing

### Combine with SFT Lora ###
# model = PeftModel.from_pretrained(
#     base_model,
#     lora_model_id,
#     token="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",
#     device_map={'': torch.cuda.current_device()},
# )
# model = model.merge_and_unload()
# model.config.use_cache = False
# model.gradient_checkpointing_enable()

### create PEFT-Lora model ###
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
# cast_mixed_precision_params(model, dtype=torch_dtype)
# print("PEFT-Lora Model created:")
# model.print_trainable_parameters()

### Load Dataset ###

## Latest UIUC.CHAT Dataset, see AICE.ipynb for detailed construction
kto_set = load_dataset("json", data_files="kto_dataset_small.json", split="train")
kto_set = kto_set.map(template_dataset)

### create accelerator ###
# accelerator_config = AcceleratorConfig(
#     split_batches=True,
#     even_batches=True,
# )

### Train KTO Model ###
# kto_config = KTOConfig(
#     # max_prompt_length=300,
#     # max_completion_length=512,
#     beta=0.1,                           # beta factor in KTO loss, Higher beta means less divergence from the initial policy.
#     desirable_weight=1.0,               # balance between good & bad answers
#     undesirable_weight=1.0,
#
#     output_dir= "./kto_log",
#     num_train_epochs=10,
#     per_device_train_batch_size=1,
#     fp16=True,
#     gradient_accumulation_steps=1,
#     gradient_checkpointing=False,
#
#     # deepspeed="./ds_config.json",
#     # accelerator_config="./fsdp_config.yaml",
# )

# kto_config = KTOConfig(
#     # script parameters
#     #model_id: "meta-llama/Meta-Llama-3-70b" # Hugging Face model id
#     #dataset_path: "./ft_dataset_uiuc.json"                      # path to dataset
#     max_prompt_length=3072, # 2048              # max sequence length for model and packing of the dataset
#     max_completion_length=3072,
#     # training parameters
#     output_dir="./llama-3-70b-uiuc",      # Temporary output directory for model checkpoints
#     #report_to: "tensorboard"               # report metrics to tensorboard
#     # learning_rate=0.0002,                  # learning rate 2e-4
#     # lr_scheduler_type="constant",          # learning rate scheduler
#     num_train_epochs=5,                   # number of training epochs
#     per_device_train_batch_size=1,         # batch size per device during training
#     per_device_eval_batch_size=1,          # batch size for evaluation
#     gradient_accumulation_steps=2,         # number of steps before performing a backward/update pass
#     # optim=adamw_torch,                     # use torch adamw optimizer
#     logging_steps=10,                      # log every 10 steps
#     # save_strategy=epoch,                   # save checkpoint every epoch
#     #evaluation_strategy=epoch,             # evaluate every epoch
#     max_grad_norm=0.3,                     # max gradient norm
#     warmup_ratio=0.03,                     # warmup ratio
#     bf16=True,                             # use bfloat16 precision
#     tf32=True,                             # use tf32 precision
#     gradient_checkpointing=True,           # use gradient checkpointing to save memory
#     beta=0.1,                           # beta factor in KTO loss, Higher beta means less divergence from the initial policy.
#     desirable_weight=1.0,               # balance between good & bad answers
#     undesirable_weight=1.0,
#     # FSDP parameters: https://huggingface.co/docs/transformers/main/en/fsdp
#     fsdp="full_shard auto_wrap offload", # remove offload if enough GPU memory
#     fsdp_config={"backward_prefetch": "backward_pre", "forward_prefetch": "false", "use_orig_params": "false"},
# )

if kto_config.gradient_checkpointing:
    model.gradient_checkpointing_enable()

kto_trainer = KTOTrainer(
    model=model,
    ref_model=model,
    args=kto_config,
    train_dataset=kto_set,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

### create accelerator ###
# accelerator = Accelerator()
# acce_trainer = accelerator.prepare(kto_trainer)
# kto_trainer = accelerator.prepare(kto_trainer)

if kto_trainer.accelerator.is_main_process:
    print("training args:", kto_config)
    print("model:", model)
    print("train_set:", kto_set)
    print("Example:")
    print("Prompt:", kto_set[0]["prompt"])
    print("Completion:", kto_set[0]["completion"])
    print("Label:", kto_set[0]["label"])
print("\nStart Training!")
kto_trainer.is_fsdp_enabled = True
kto_trainer.train()

### Save Model ###
# accelerator.wait_for_everyone()
# # kto_trainer.model.save_pretrained("./Mistral_8x7B_SFT_KTO_Lora")
# # kto_trainer.tokenizer.save_pretrained("./Mistral_8x7B_SFT_KTO_Lora")
# if accelerator.is_main_process:
#     unwrapped_model = accelerator.unwrap_model(kto_trainer.model)
#     unwrapped_tokenizer = accelerator.unwrap_model(kto_trainer.tokenizer)
#     unwrapped_model.push_to_hub("LongQ/Llama3_UIUC_SFT_KTO_Lora", private=False)
#     unwrapped_tokenizer.push_to_hub("LongQ/Llama3_UIUC_SFT_KTO_Lora", private=False)

# kto_trainer.is_fsdp_enabled = True

# kto_trainer.accelerator.wait_for_everyone()
kto_trainer.save_model()

print("Pushing Models to Hub...")
kto_trainer.model.push_to_hub("LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_KTO_Lora", private=True, token="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")
kto_trainer.tokenizer.push_to_hub("LongQ/Llama3_70B_CHAT_UIUC_ECE_SFT_KTO_Lora", private=True, token="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")

# if kto_trainer.accelerator.is_main_process:
#     print("Pushing Models to Hub...")
#     kto_trainer.model.push_to_hub("LongQ/Llama3_8B_CHAT_UIUC_ECE_SFT_KTO_Lora", private=True, token="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")
#     kto_trainer.tokenizer.push_to_hub("LongQ/Llama3_8B_CHAT_UIUC_ECE_SFT_KTO_Lora", private=True, token="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")
#     print("Successfully pushed model.")
# kto_trainer.accelerator.wait_for_everyone()

### Run with Accelerator ###
## accelerate launch --config_file "./fsdp_config.yaml" KTO_acce.py
## accelerate launch --config_file "./config.yaml" KTO_acce.py

print("Exit Normally.")



