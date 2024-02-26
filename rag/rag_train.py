import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, RagRetriever, RagModel, RagTokenForGeneration
# import faiss
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset

### Try GPU ###
try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")

### Train model end-to-end ###
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
# model = RagTokenForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
model.to(device)

# ##### Truthful QA #####
# ### Build Dataset -- Truthful QA ###
# from datasets import load_dataset
# dataset = load_dataset("truthful_qa", "generation")
# dataset = dataset["validation"]
#
# ## Make training set
# train_inputs = []
# train_labels = []
# for i in range(400, 750):
#   train_inputs.append(dataset[i]['question'])
#   train_labels.append(dataset[i]['best_answer'])
#
# ## Make validation set
# valid_inputs = []
# valid_labels = []
# for j in range(750, 800):
#   valid_inputs.append(dataset[j]['question'])
#   valid_labels.append(dataset[j]['best_answer'])
#
# print("Loading dataset finished : {} training set, {} validation set.".format(len(train_inputs), len(valid_inputs)))
# print(dataset[0])
#
# ### Rough train ###
# print("Start Training!")
# model.to(device)
# model.train()
#
# from transformers.optimization import Adafactor
#
# optimizer = Adafactor(
#     model.parameters(),
#     lr=1e-3,
#     eps=(1e-30, 1e-3),
#     clip_threshold=1.0,
#     decay_rate=-0.8,
#     beta1=None,
#     weight_decay=0.0,
#     relative_step=False,
#     scale_parameter=False,
#     warmup_init=False,
# )
#
# ## Run Model
# epoches = 20
#
# for epoch in range(epoches):
#     totalLoss = 0.0
#     for idx in range(len(train_inputs)):
#         inputs = tokenizer(train_inputs[idx], return_tensors="pt").to(device)
#         targets = tokenizer(train_labels[idx], return_tensors="pt").to(device)
#         input_ids = inputs["input_ids"]
#         labels = targets["input_ids"]
#         outputs = model(input_ids=input_ids, labels=labels)
#
#         if idx % 50 == 0:
#             print("Sample input {}: \n".format(idx + 1), train_inputs[idx])
#             print("Sample output {}: \n".format(idx + 1), tokenizer.decode((model.generate(input_ids=input_ids))[0], skip_special_tokens=True))
#             print("Sample gold {}: \n".format(idx + 1), train_labels[idx])
#             print("\n")
#
#         loss = outputs.loss.to(device)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         totalLoss = totalLoss + loss.item()
#     print("Epoch {} : Loss {}.\n".format(epoch + 1, totalLoss))
#
# model.save_pretrained("./rag_model")
# retriever.save_pretrained("./rag_retriever")
# print("Training finished.")
#
# ### Validate Model ###
# model.eval()
# totalLossVal = 0.0
# with torch.no_grad():
#     for idx in range(len(valid_inputs)):
#         inputs = tokenizer(valid_inputs[idx], return_tensors="pt").to(device)
#         targets = tokenizer(valid_labels[idx], return_tensors="pt").to(device)
#         input_ids = inputs["input_ids"]
#         labels = targets["input_ids"]
#
#         if idx % 10 == 0:
#             print("Sample Val input {}: \n".format(idx + 1), valid_inputs[idx])
#             print("Sample Val output {}: \n".format(idx + 1), tokenizer.decode((model.generate(input_ids=input_ids))[0]))
#             print("Sample Val gold {}: \n".format(idx + 1), valid_labels[idx])
#             print("\n")
#         loss = model(input_ids=input_ids, labels=labels).loss.to(device)
#         totalLossVal = totalLossVal + loss.item()
# print("Validation result : Loss {}".format(totalLossVal))

# ##### WikiQA #####
#
# ### Huggingface Trainer ###
#
# ## build torch.dataset
# dataset = load_dataset("wiki_qa")
# print(dataset)
#
# training_args = TrainingArguments(
#     output_dir="./arg_model_wikiqa",
#     learning_rate=1e-3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     tokenizer=tokenizer,
# )
#
# print("Start Training!")
# trainer.train()