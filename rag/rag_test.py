from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
import torch
from datasets import load_dataset

# ### Try GPU
# try:
#     device = torch.device("cuda")
# except:
#     device = torch.device("cpu")

### Load Model

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("./rag_retriever", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("./rag_model")

### Load dataset
dataset = load_dataset("truthful_qa", "generation")
dataset = dataset["validation"]

## Make training set
questions = []
for i in range(50):
    questions.append(dataset[i]['question'])

### Test RAG retriever

with torch.no_grad():

    for idx in range(10):
        inputs = tokenizer(questions[idx], return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Step 1: Encode Query
        question_hidden_states = model.question_encoder(input_ids)[0]

        # Step 2: Retrieve Doc
        docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

        # Step 3: Output Match Score and retrieved doc
        print("Case {}:".format(idx))
        print(doc_scores)
        print(docs_dict['doc_ids'])
        doc_id = docs_dict['doc_ids']
        context = retriever.index.get_doc_dicts(doc_id)
        print(context[0]['text'][0])