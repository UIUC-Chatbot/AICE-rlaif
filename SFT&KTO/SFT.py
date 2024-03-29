import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from accelerate import PartialState
import os
from peft import LoraConfig, TaskType, get_peft_model
from random import shuffle

### PEFT setup ###
peft_config = LoraConfig(task_type="CAUSAL_LM",
                         inference_mode=False,
                         r=8,
                         lora_alpha=32,
                         lora_dropout=0.1)

### Try gpu ###
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print(torch.cuda.device_count())

### Base Model ###
model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory = { 0: '30.0GiB', 1: '30.0GiB',})
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

### Transform to PEFT model
model = get_peft_model(model, peft_config)

### Build Dataset : See Jupyter Notebook & ft_dataset.json ###

# Use 300 samples from TruthfulQA for training
data_start = 100
data_end = 400
train_questions = load_dataset("truthful_qa", "generation")["validation"][data_start:data_end]["question"]
train_answers = load_dataset("truthful_qa", "generation")["validation"][data_start:data_end]["best_answer"]

# with open("ft_dataset.json", "w", encoding='utf-8') as f:
#     for idx in range(len(train_questions)):
#         f.write(json.dumps({"prompt": train_questions[idx], "completion": train_answers[idx]}))
#         f.write("\n")

train_set = load_dataset("json", data_files="ft_dataset.json", split="train")
# train_set = load_dataset("LongQ/TruthfulQA_critique_small", split="train")

## Build Training Samples

def trainable_dataset(dataset):
    output_texts = []
    for i in range(len(dataset['prompt'])):
        text = dataset["prompt"][i] + "\nThe answer is " + dataset["completion"][i]
        output_texts.append(text)
    return output_texts

### Training arguments ###

training_arguments = TrainingArguments(
    output_dir= "./finetune_log",
    num_train_epochs= 2,
    auto_find_batch_size=True,
)

### Trainer ###
device_string = PartialState().process_index

trainer = SFTTrainer(
    model,
    train_dataset=train_set,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=trainable_dataset,
)

### Train ###
trainer.train()

### Save model ###
trainer.model.save_pretrained("Mistral_SFT_TFQA")
trainer.tokenizer.save_pretrained("Mistral_SFT_TFQA_tokenizer")
trainer.model.push_to_hub("LongQ/Mistral_SFT_TFQA", private=True, token="")
trainer.tokenizer.push_to_hub("LongQ/Mistral_SFT_TFQA", private=True, token="")
# model = AutoModelForCausalLM.from_pretrained("Mistral_SFT_100")
# model.push_to_hub("LongQ/Mistral_SFT_100", private=True, token="")