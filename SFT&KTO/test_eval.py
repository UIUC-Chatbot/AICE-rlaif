from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

### Prepare dataset ###

samples_start = 0
samples_end = 100

### Truthful QA ###
test_dataset = load_dataset("truthful_qa", "generation")["validation"][samples_start:samples_end]
questions = test_dataset["question"]
responses = []
gold_best = test_dataset["best_answer"]
gold_correct = test_dataset["correct_answers"]
gold_incorrect = test_dataset["incorrect_answers"]

print("Sample 0:")
print(questions[0])
print(gold_best[0])
print(gold_correct[0])
print(gold_incorrect[0])
print("Finish building dataset of {} samples.\n".format(samples_end - samples_start))

### Base Model Evaluation ###
df_test = pd.DataFrame()
df_test['question'] = ''
df_test['response'] = ''
df_test['gold_best'] = ''
df_test['gold_correct'] = ''
df_test['gold_incorrect'] = ''
df_test['gold_best_score'] = ''
df_test['gold_correct_score'] = ''
df_test['gold_incorrect_score'] = ''
df_test['class'] = ''
df_testRow = 0

### Load Model ###
# model_id = "LongQ/Mistral_SFT_TFQA_collator"
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "LongQ/Mistral_KTO_Lora_TFQA"
# model_id = "LongQ/Mistral_SFT_Lora_TFQA"
# model_id = "LongQ/Mistral_SFT_TFQA"
model_id = "LongQ/Mistral_SFT_KTO"
# model_id = "LongQ/Mistral_KTO_CRISMALL"

tokenizer = AutoTokenizer.from_pretrained(model_id, token="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", padding_side="left")

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory = { 0: '30.0GiB', 1: '30.0GiB',}, token="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")

### Generating Response ###
for idx in range(len(questions)):
    text = questions[idx] + "\nThe answer is "
    print("Generating answer for question :", idx)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to("cuda")
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device="cuda")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
    )
    # ## delete questions
    outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    responses.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    df_test.loc[idx, 'question'] = questions[idx]
    df_test.loc[idx, 'answer'] = responses[idx]
    df_test.loc[idx, 'gold_best'] = gold_best[idx]

print("Sample 0:")
print(responses[0])
print("\nBase model finished generating {} responses.\n".format(len(questions)))

# ## Test Teacher Pipe
# train_dataset = load_dataset("LongQ/TruthfulQA_critique_small", split="train")
# questions = train_dataset["question"][0:100]
# responses = train_dataset["final_answer"][0:100]

### Evaluation using token similarity ###

model_eval = SentenceTransformer("all-MiniLM-L6-v2")

match_best = 0
match_correct = 0
match_incorrect = 0
match_book = []     # list of dict of matching records

for idx in range(len(questions)):
    if idx % 50 == 0:
        print("Evaluating response {}".format(idx))

    match_record = {}
    match_record["question"] = questions[idx]
    match_record["response"] = responses[idx]
    score_max = 0

    # Step 1: match best
    sentences1 = responses[idx]
    sentences2 = gold_best[idx]
    embeddings1 = model_eval.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_eval.encode(sentences2, convert_to_tensor=True)
    scores = util.cos_sim(embeddings1, embeddings2)[0]
    score_max = scores.tolist()[0]       # similarity value of best
    match_record["best"] = sentences2
    df_test.loc[df_testRow, 'gold_best_score'] = score_max
    result = "best"

    # Step 2: match one of gold_correct
    sentences2 = gold_correct[idx]
    embeddings1 = model_eval.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_eval.encode(sentences2, convert_to_tensor=True)
    scores = util.cos_sim(embeddings1, embeddings2)[0]
    scores = scores.tolist()
    correct = max(scores)       # similarity value of most match gold_correct
    match_id = scores.index(correct)
    match_record["correct"] = sentences2[match_id]
    df_test.loc[df_testRow, 'gold_correct'] = sentences2[match_id]
    df_test.loc[df_testRow, 'gold_correct_score'] = correct
    if correct > score_max:
        score_max = correct
        result = "correct"

    # Step 3: match one of gold_incorrect
    sentences2 = gold_incorrect[idx]
    embeddings1 = model_eval.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_eval.encode(sentences2, convert_to_tensor=True)
    scores = util.cos_sim(embeddings1, embeddings2)[0]
    scores = scores.tolist()
    incorrect = max(scores)       # similarity value of most match gold_correct
    match_id = scores.index(incorrect)
    match_record["incorrect"] = sentences2[match_id]
    df_test.loc[df_testRow, 'gold_incorrect'] = sentences2[match_id]
    df_test.loc[df_testRow, 'gold_incorrect_score'] = incorrect
    if incorrect > score_max:
        score_max = incorrect
        result = "incorrect"

    # Step 4: classify the case
    if result == "best":
      match_best += 1
    elif result == "correct":
      match_correct += 1
    elif result == "incorrect":
      match_incorrect += 1
    match_record["result"] = result
    match_record["confidence"] = score_max
    df_test.loc[df_testRow, 'class'] = result
    match_book.append(match_record)

    df_testRow += 1

## visualize some results
print("Best: Number = {}, frequency = {}".format(match_best, match_best/len(questions)))
print("Correct: Number = {}, frequency = {}".format(match_correct, match_correct/len(questions)))
print("Incorrect: Number = {}, frequency = {}".format(match_incorrect, match_incorrect/len(questions)))

print("\nEvaluation of {} responses finished.\n".format(len(responses)))

## save to excel
df_test.to_excel('./test_SFT_KTO.xlsx', sheet_name='Sheet1', index=False)
