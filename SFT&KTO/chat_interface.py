import torch
import transformers
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from datasets import Dataset
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

### New Version ###

### Load datasets
from datasets import load_dataset

dataset = load_dataset("CAII-NCSA/ECE_RAG_critique_judge", token="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")
train_set = dataset["train"]

print(train_set)

### System message
system_message = """
Please analyze and respond to the following question using the excerpts from the provided documents. These documents can be pdf files or web pages. Additionally, you may see the output from API calls (calld 'tools') to the user's services which, when relevant, you should use to construct your answer. You may also see image descriptions from images uploaded by the user. Prioritize image descriptions, when helpful, to construct your answer.

Integrate relevant information from these documents, ensuring each reference is linked to the document's number.

Your response should be semi-formal.

When quoting directly, cite with footnotes linked to the document number and page number, if provided.

Summarize or paraphrase other relevant information with inline citations, again referencing the document number and page number, if provided.

If the answer is not in the provided documents, state so. Yet always provide as helpful a response as possible to directly answer the question.

Conclude your response with a LIST of the document titles as clickable placeholders, each linked to its respective document number and page number, if provided.

Always share page numbers if they were shared with you.

ALWAYS follow the examples below:

Insert an inline citation like this in your response:

"[1]" if you're referencing the first document or

"[1, page: 2]" if you're referencing page 2 of the first document.

At the end of your response, list the document title with a clickable link, like this:

"1. [document_name](#)" if you're referencing the first document or

"1. [document_name, page: 2](#)" if you're referencing page 2 of the first document.

Nothing else should prefixxed or suffixed to the citation or document name. Consecutive citations should be separated by a comma.



Suppose a document name is shared with you along with the index and pageNumber below like "27: www.pdf, page: 2", "28: www.osd", "29: pdf.www, page 11\n15" where 27, 28, 29 are indices, www.pdf, www.osd, pdf.www are document_name, and 2, 11 are the pageNumbers and 15 is the content of the document, then inline citations and final list of cited documents should ALWAYS be in the following format:

\"\"\"

The sky is blue. [27, page: 2][28] The grass is green. [29, page: 11]

Relevant Sources:



27. [www.pdf, page: 2](#)

28. [www.osd](#)

29. [pdf.www, page: 11](#)

\"\"\"

ONLY return the documents with relevant information and cited in the response. If there are no relevant sources, don't include the "Relevant Sources" section in response.

The following are excerpts from the high-quality documents, APIs/tools, and image descriptions to construct your answer.\n
"""

# print(system_message)

### User message
sample = {}
sample["RAG"] = train_set[0]["RAG"]
sample["question"] = train_set[0]["question"]
# print(sample)

user_message = """
  Here's high quality passages from the user's documents. Use these, and cite them carefully in the format previously described, to construct your answer:
  {}
  Finally, please respond to the user's query: {}
""".format(sample["RAG"], sample["question"])
print("user_message :\n", user_message)

### Generate response
from transformers import pipeline
import torch

model_id = "CAII-NCSA/Llama3_70B_CHAT_UIUC_ECE_SFT_Lora_Merged"

pipe = pipeline(
    "text-generation",
    model=model_id,
    token="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",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]

LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
def template_dataset(examples):
    return {"text": pipe.tokenizer.apply_chat_template(examples, tokenize=False)}

text = template_dataset(messages)

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
)
print("ChatBot's Answer :\n", outputs[0]["generated_text"][-1])




