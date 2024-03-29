from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model

### Try gpu ###
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print(torch.cuda.device_count())

### Build Dataset ###
tf_dataset = load_dataset("truthful_qa", "generation")["validation"][0:100]
train_dataset = load_dataset("LongQ/TruthfulQA_critique_small", split="train")

prompts = []
completions = []
labels = []

for q_idx in range(len(train_dataset["question"])):

    ## store best answer
    prompts.append(train_dataset["question"][q_idx])
    completions.append("The answer is" + train_dataset["final_answer"][q_idx])
    labels.append(True)

for q_idx in range(len(tf_dataset["question"])):

    ## store best answer
    # prompts.append(tf_dataset["question"][q_idx])
    # completions.append("The answer is " + tf_dataset["best_answer"][q_idx])
    # labels.append(True)

    # ## store correct answer
    # for corr_idx in range(len(train_dataset["correct_answers"][q_idx])):
    #     prompts.append(train_dataset["question"][q_idx])
    #     completions.append(train_dataset["correct_answers"][q_idx][corr_idx])
    #     labels.append(True)

    ## store incorrect answer
    for incorr_idx in range(len(tf_dataset["incorrect_answers"][q_idx])):
        prompts.append(tf_dataset["question"][q_idx])
        completions.append("The answer is" +tf_dataset["incorrect_answers"][q_idx][incorr_idx])
        labels.append(False)

print("Number of samples :", len(prompts))

kto_dataset_dict = {"prompt":prompts, "completion":completions, "label":labels}
train_dataset = Dataset.from_dict(kto_dataset_dict)

### Load Model ###
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "LongQ/Mistral_SFT_TFQA"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token = True

model = AutoModelForCausalLM.from_pretrained(model_id, token="", device_map="auto", max_memory={0: '30.0GiB', 1: '30.0GiB'})

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

### Train KTO Model ###
training_args = KTOConfig(
    output_dir= "./finetune_log",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    num_train_epochs=10,
    per_gpu_train_batch_size=16,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
)

kto_trainer = KTOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("\nStart Training!")
kto_trainer.train()

### Save Model ###
# kto_trainer.model.save_pretrained("Mistral_KTO_TFQA")
# kto_trainer.tokenizer.save_pretrained("Mistral_KTO_TFQA")
kto_trainer.model.push_to_hub("LongQ/Mistral_SFT_KTO", private=True, token="")
kto_trainer.tokenizer.push_to_hub("LongQ/Mistral_SFT_KTO", private=True, token="")