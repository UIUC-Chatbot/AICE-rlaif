from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from accelerate import Accelerator, PartialState
import pandas as pd

### Try gpu ###
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

### accelerator
accelerator = Accelerator()

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

## UIUC.CHAT Dataset
data_file = "./final_judge.csv"
data_csv = pd.read_csv(data_file)

questions = data_csv['question'].to_list()
answers = data_csv['answer'].to_list()
scores = data_csv['llm_judge_score'].to_list()

# Data Cleaning
idx = 0
while idx < len(questions):
    if type(questions[idx]) != str or type(answers[idx]) != str or type(scores[idx]) != float or scores[idx] > 10 or scores[idx] < 0 or scores[idx] != scores[idx]:
        del questions[idx]
        del answers[idx]
        del scores[idx]
        continue
    else:
        idx += 1

# Select only score>=7
questions_best = []
answers_best = []
for idx in range(len(questions)):
    if scores[idx] >= 7:
        questions_best.append(questions[idx])
        answers_best.append(answers[idx])

print("Filter best dataset of length :", len(questions_best))

## Select only score<=3
questions_bad = []
answers_bad = []
for idx in range(len(questions)):
    if scores[idx] <= 3:
        questions_bad.append(questions[idx])
        answers_bad.append(answers[idx])

print("Filter bad dataset of length :", len(questions_bad))

## Labeling Dataset
prompts = []
completions = []
labels = []

for q_idx in range(len(questions_best)):

    ## store best answer
    prompts.append("Question:\n" + questions_best[q_idx] + "\nAnswer:")
    completions.append(answers_best[q_idx])
    labels.append(True)

for q_idx in range(len(questions_bad)):

    ## store best answer
    prompts.append("Question:\n" + questions_bad[q_idx] + "\nAnswer:")
    completions.append(answers_bad[q_idx])
    labels.append(False)

print("Number of samples :", len(prompts))

kto_dataset_dict = {"prompt":prompts, "completion":completions, "label":labels}
train_dataset = Dataset.from_dict(kto_dataset_dict)

### Load Model ###

### Quantization ###
quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
)

# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "LongQ/Mistral_SFT_TFQA"

# lora_model_id = "LongQ/Mistral_8x7B_SFT_Lora"
lora_model_id = "LongQ/Llama3_UIUC_SFT_Lora"

tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.add_eos_token = True
tokenizer.add_bos_token = True

# base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: '30.0GiB', 1: '30.0GiB'})
# base_model = AutoModelForCausalLM.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    max_memory = { 0: '30.0GiB', 1: '30.0GiB'},
    # quantization_config=quantization_config,
    # torch_dtype=torch.float16,
    device_map={'':torch.cuda.current_device()},
)

### Combine with SFT Lora ###
model = PeftModel.from_pretrained(base_model, lora_model_id, token="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")
model = model.merge_and_unload()
# model = base_model
# model.config.use_cache = False

### create PEFT-Lora model ###
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
)
model = get_peft_model(model, peft_config)
print("PEFT-Lora Model created:")
model.print_trainable_parameters()

### create accelerator ###
# accelerator_config = AcceleratorConfig(
#     split_batches=True,
#     even_batches=True,
# )

### Train KTO Model ###
kto_config = KTOConfig(
    # max_prompt_length=300,
    # max_completion_length=512,
    beta=0.1,                           # beta factor in KTO loss, Higher beta means less divergence from the initial policy.
    desirable_weight=1.0,               # balance between good & bad answers
    undesirable_weight=1.0,

    output_dir= "./kto_log",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    fp16=True,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,

    # deepspeed="./ds_config.json",
    # accelerator_config="./fsdp_config.yaml",
)

kto_trainer = KTOTrainer(
    model=model,
    ref_model=model,
    args=kto_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

### create accelerator ###
# accelerator = Accelerator()
# acce_trainer = accelerator.prepare(kto_trainer)
kto_trainer = accelerator.prepare(kto_trainer)

print("\nStart Training!")
kto_trainer.train()

### Save Model ###
accelerator.wait_for_everyone()
# kto_trainer.model.save_pretrained("./Mistral_8x7B_SFT_KTO_Lora")
# kto_trainer.tokenizer.save_pretrained("./Mistral_8x7B_SFT_KTO_Lora")
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(kto_trainer.model)
    unwrapped_tokenizer = accelerator.unwrap_model(kto_trainer.tokenizer)
    unwrapped_model.push_to_hub("LongQ/Llama3_UIUC_SFT_KTO_Lora", private=False)
    unwrapped_tokenizer.push_to_hub("LongQ/Llama3_UIUC_SFT_KTO_Lora", private=False)

### Run with Accelerator ###
## accelerate launch --config_file "./fsdp_config.yaml" KTO_acce.py
## accelerate launch --config_file "./config.yaml" KTO_acce.py





