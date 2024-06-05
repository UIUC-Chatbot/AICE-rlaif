from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from trl.commands.cli_utils import TrlParser
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, AutoPeftModelForCausalLM
from accelerate import Accelerator, PartialState
from dataclasses import dataclass, field
import pandas as pd
import yaml

## Load Model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_model_id = "LongQ/Llama3_8B_CHAT_UIUC_ECE_SFT_Lora"

torch_dtype = torch.bfloat16
quant_storage_dtype = torch.bfloat16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=quant_storage_dtype,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
    torch_dtype=torch_dtype,
    use_cache=False,  # this is needed for gradient checkpointing
    # device_map={'':torch.cuda.current_device()},
)

model = PeftModel.from_pretrained(
    base_model,
    lora_model_id,
    token="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",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
    device_map={'': torch.cuda.current_device()},
)

# model = AutoPeftModelForCausalLM.from_pretrained(
#     lora_model_id,
#     token="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",
#     quantization_config=quantization_config,
#     attn_implementation="sdpa",
#     torch_dtype=quant_storage_dtype,
#     use_cache=False,
#     # device_map={'':torch.cuda.current_device()},
# )

merged_model = model.merge_and_unload()
print("Pushing Models to Hub...")
merged_model.push_to_hub("LongQ/Llama3_8B_CHAT_UIUC_ECE_SFT_Lora_Merged", private=True, token="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")
print("Exit Normally")








