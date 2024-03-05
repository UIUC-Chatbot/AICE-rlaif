from datasets import load_dataset
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer, util

### API key ###
client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = ""
)

### Prepare dataset ###

dataset = load_dataset("truthful_qa", "generation")
dataset = dataset["validation"]
print(len(dataset))
print(dataset[0])
print("Dataset Loading finished.\n")

questions = []
responses = []    # model predictions
gold_best = []    # single optimal gold answer
gold_correct = []     # list of lists
gold_incorrect = []
sample_num = 50

for idx in range(sample_num):
    questions.append(dataset[idx]["question"])
    gold_best.append(dataset[idx]["best_answer"])
    gold_correct.append(dataset[idx]["correct_answers"])
    gold_incorrect.append(dataset[idx]["incorrect_answers"])

print("Sample 0:")
print(questions[0])
print(gold_best[0])
print(gold_correct[0])
print(gold_incorrect[0])
print("Finish building dataset of {} samples.\n".format(sample_num))

### Base Model Evaluation ###
df_base = pd.DataFrame()
df_base['idx'] = ''
df_base['question'] = ''
df_base['response'] = ''
df_base['gold_best'] = ''
df_base['gold_correct'] = ''
df_base['gold_incorrect'] = ''
df_base['gold_best_score'] = ''
df_base['gold_correct_score'] = ''
df_base['gold_incorrect_score'] = ''
df_base['class'] = ''
df_baseRow = 0

## Generate Model Prediction
for idx in range(len(questions)):

    completion = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=questions[idx],
        max_tokens=30,
        temperature = 0.5
    )
    response = completion.choices[0].text.strip()

    responses.append(response)
    df_base.loc[df_baseRow, 'idx'] = idx
    df_base.loc[df_baseRow, 'question'] = questions[idx]
    df_base.loc[df_baseRow, 'response'] = response
    df_base.loc[df_baseRow, 'gold_best'] = gold_best[idx]

    df_baseRow += 1

print("Sample 0:")
print(responses[0])
print("\nBase model finished generating {} responses.\n".format(len(questions)))

## Evaluation using token similarity

df_baseRow = 0

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
    match_record["response"] = responses[idx][0]
    score_max = 0

    # Step 1: match best
    sentences1 = responses[idx]
    sentences2 = gold_best[idx]
    embeddings1 = model_eval.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_eval.encode(sentences2, convert_to_tensor=True)
    scores = util.cos_sim(embeddings1, embeddings2)[0]
    score_max = scores.tolist()[0]       # similarity value of best
    match_record["best"] = sentences2
    df_base.loc[df_baseRow, 'gold_best_score'] = score_max
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
    df_base.loc[df_baseRow, 'gold_correct'] = sentences2[match_id]
    df_base.loc[df_baseRow, 'gold_correct_score'] = correct
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
    df_base.loc[df_baseRow, 'gold_incorrect'] = sentences2[match_id]
    df_base.loc[df_baseRow, 'gold_incorrect_score'] = incorrect
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
    df_base.loc[df_baseRow, 'class'] = result
    match_book.append(match_record)

    df_baseRow += 1

## visualize some results
print("Best: Number = {}, frequency = {}".format(match_best, match_best/len(questions)))
print("Correct: Number = {}, frequency = {}".format(match_correct, match_correct/len(questions)))
print("Incorrect: Number = {}, frequency = {}".format(match_incorrect, match_incorrect/len(questions)))

print("\nEvaluation of {} responses finished.\n".format(len(responses)))

## save to excel
df_base.to_excel('./base_eval6.xlsx', sheet_name='Sheet1', index=False)
