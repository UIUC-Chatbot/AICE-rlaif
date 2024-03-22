from huggingface_hub import login
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers import AlbertModel, AlbertTokenizer
from peft import PeftModel
from utils import process_RAG_data, ask_question_no_generator
import torch

passage_titles = []
passages = []
test_question = "What's the difference between current RAG framework and self-RAG"

# Load data
with open ("self_rag.txt", 'r') as f:
    text = f.read()
    words = text.split()
    for i in range(0, len(words), 100): # Loop over the words, incrementing by 100
        chunk_words = words[i: i+100]
        chunk = " ". join(chunk_words)
        chunk = chunk.strip()
        if len(chunk) == 0: # Skip empty chunks
            continue
        passage_titles.append(chunk[0]) # Only one paper, no title. So I use the first letter as title
        passages.append(chunk)

corpus = {'title':passage_titles, 'text': passages}
corpus_dataset, index = process_RAG_data(corpus)
titles, passages = ask_question_no_generator(test_question, corpus, index)

login(
token="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", # ADD YOUR TOKEN HERE
add_to_git_credential=True
)


tokenizer = AutoTokenizer.from_pretrained("ShuyiGuo/self-critique-RAG")
model = AutoPeftModelForCausalLM.from_pretrained(
  "ShuyiGuo/self-critique-RAG",
  device_map="auto",
  torch_dtype=torch.float16
)
#tokenizer = AutoTokenizer.from_pretrained("ShuyiGuo/self-critique-RAG")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the context and the question

context = f"There're some references. {passages[0]} You need to evaluate whether they're relevant to the following question, and answer the question.{test_question}"

outputs = pipe(context, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
print("This is with RAG")
print(outputs)
# prompt = pipe.tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
# print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

print("This is without RAG")
outputs = pipe(test_question, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
print(outputs)