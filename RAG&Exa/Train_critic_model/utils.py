import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import DPRContextEncoderTokenizerFast
from transformers import DPRContextEncoder
from transformers import RagRetriever, RagTokenizer, RagSequenceForGeneration
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import json
import openai
import time
import datetime
import math
import faiss
import pickle
import torch
import textwrap

def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds = elapsed_rounded))

def tokenize(corpus, tokenizer = "facebook/dpr-ctx_encoder-multiset-base"):
  ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(tokenizer)

  outputs = ctx_tokenizer(
      corpus['title'],
      corpus['text'],
      truncation = True,
      padding = 'longest',
      return_tensors = 'pt'
  )

  print('------Tokenization Done------')
  input_ids = outputs['input_ids']

  return input_ids

def encode(input_ids, corpus, encoder = "facebook/dpr-ctx_encoder-multiset-base", batch_size = 16):
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
  else:
    print("No GPU is available")
  # Encode
  ctx_encoder = DPRContextEncoder.from_pretrained(encoder)

  # move the encoder model to the GPU
  ctx_encoder = ctx_encoder.to(device = device)

  # no need for gradient descent
  torch.set_grad_enabled(False)

  # Track elapsed time & the current batch number for progress updates
  t0 = time.time()
  step = 0
  batch_size = batch_size

  num_passages = input_ids.size()[0]
  num_batches = math.ceil(num_passages / batch_size)

  # Accumulate embedded passages in the list
  #SH_embeds_batches = []
  embeds_batches = []

  print('Generating embeddings for {:,} passages...'.format(num_passages))

  for i in range(0, num_passages, batch_size):
    if step % 100 == 0 and not step == 0: # update every 100 batches
      elapsed = format_time(time.time() - t0)
      print("Batch{:>5} of {:>5}. Elasped:{:}.".format(step, num_batches, elapsed))

    # Select the next batch and move them to the GPU
    batch_ids = input_ids[i:i+16, :]
    batch_ids = batch_ids.to(device)

    # Run the encoder
    outputs = ctx_encoder(
        batch_ids,
        return_dict = True
    )

    embeddings = outputs["pooler_output"]

    # Bring the embeddings back over from the GPU and convert to numpy (out of pytorch)
    embeddings = embeddings.detach().cpu().numpy()

    embeds_batches.append(embeddings)

    step += 1

  print("------DONE------")
  embeddings = np.concatenate(embeds_batches, axis=0)

  # Adjust the dict into Dataset object
  corpus_df = pd.DataFrame(corpus)
  corpus_dataset = Dataset.from_pandas(corpus_df)


  embeddings_list_array = []
  for i in range(embeddings.shape[0]):
    embeddings_list_array.append(embeddings[i,:])

  corpus_dataset = corpus_dataset.add_column("embeddings", embeddings_list_array)

  return corpus_dataset

def Faiss_index(corpus_dataset, dim = 768, m = 128):
  # Faiss index
  dim = dim
  m = m

  # Use the Faiss implementation of HNSW for fast approximate nearest neighbor search
  index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
  corpus_dataset.add_faiss_index(column = "embeddings", index_name = "embeddings", custom_index = index, faiss_verbose = True)

  return corpus_dataset, index

def process_RAG_data(corpus, tokenizer = "facebook/dpr-ctx_encoder-multiset-base", encoder = "facebook/dpr-ctx_encoder-multiset-base", batch_size = 16, dim = 768, m = 128):
  input_ids = tokenize(corpus, tokenizer = "facebook/dpr-ctx_encoder-multiset-base")
  corpus_dataset = encode(input_ids, corpus, encoder = "facebook/dpr-ctx_encoder-multiset-base", batch_size = 16)
  corpus_dataset, index = Faiss_index(corpus_dataset)

  return corpus_dataset, index

def prepare_model(corpus_dataset, retrieve_model = "facebook/rag-sequence-nq", tokenize_model = "facebook/rag-sequence-nq", generate_model = "facebook/rag-sequence-nq", Use_dummy_dataset = False):
  retriever = RagRetriever.from_pretrained(
      retrieve_model,
      use_dummy_dataset = False,
      indexed_dataset = corpus_dataset, # Use our own dataset
      index_name = "embeddings_index")
  tokenizer = RagTokenizer.from_pretrained(tokenize_model)
  model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever = retriever)

  return retriever, tokenizer, model

def ask_question(question, tokenizer, model):
  t0 = time.time()

  input_ids = tokenizer.question_encoder(question, return_tensors = "pt")['input_ids']
  generated = model.generate(input_ids)
  generated_str = tokenizer.batch_decode(generated, skip_special_tokens = True)[0]

  print("Q: " + question)
  print("A: '{:}'". format(generated_str))

  print("\nResponse took %.2f seconds" %(time.time() - t0))

def question_tokenize_and_encode(question, 
                                 encode_model = "facebook/dpr-question_encoder-multiset-base", 
                                 tokenize_model = "facebook/dpr-question_encoder-multiset-base" ):
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
  else:
    print("No GPU is available")

  q_encoder = DPRQuestionEncoder.from_pretrained(encode_model)
  q_encoder = q_encoder.to(device = device)   # Move the encoder model to the GPU
  q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(tokenize_model)

  input_ids = q_tokenizer.encode(question, return_tensors = "pt") # Tokenize the question
  input_ids = input_ids.to(device)  # Move the question to GPU

  # Run the question through BERT and generate the question embedding
  outputs = q_encoder(input_ids)
  q_embed = outputs['pooler_output']

  # Transfer the embeddings to our CPU to do the research
  q_embed = q_embed.detach().cpu().numpy()

  return q_embed

def retrieve(q_embed, index, corpus, k = 3, width = 80):
  # Perform search using the FAISS index
  # Find the k=3 most similar passages to the question embedding
  D, I = index.search(q_embed, k)
  print("Closest matching indeces:", I)
  print("Inner Products:", D)

  # Map the indeces back to the original passage text
  # Wrap text to 80 characters
  titles = []
  passages = []
  wrapper = textwrap.TextWrapper(width)
  for i in I[0]:
    print("Index:", i)
    title = corpus["title"][i]
    titles.append(title)
    passage = corpus["text"][i]
    passages.append(passage)
    print("Article Title: ", title, "\n")
    print("Passages: ")
    print(wrapper.fill(passage))

  return titles, passages

def ask_question_no_generator(question, corpus, index,
                              encode_model = "facebook/dpr-question_encoder-multiset-base",
                              tokenize_model = "facebook/dpr-question_encoder-multiset-base",
                              k = 3,
                              width = 80):
  q_embed = question_tokenize_and_encode(question, encode_model, tokenize_model)
  titles, passages = retrieve(q_embed, index, corpus, k, width)
  return titles, passages

system_message = """You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional).
If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.
Your task is to determine whether the information in the output sentence can be fully verified by the evidence
or if it requires further external verification.
There are three cases:
- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence].
- If the sentence doesn’t require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [No Retrieval].
- If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments."""

def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": sample["instruction"]},
      {"role": "assistant", "content": sample["output"]}
    ]
  }