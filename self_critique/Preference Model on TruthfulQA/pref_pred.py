from datasets import load_dataset
import random
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer, util
import json

### Load data ###

data = pd.read_excel("./base_all.xlsx")
questions = data["question"].tolist()
responses1 = data["response1"].tolist()
responses2 = data["response2"].tolist()
responses3 = data["response3"].tolist()
responses4 = data["response4"].tolist()

prompts = []             # for prediction

for idx in range(len(questions)):
    answers = ""

    answers += "{}:".format(1) + str(responses1[idx]) + "; "
    answers += "{}:".format(2) + str(responses2[idx]) + "; "
    answers += "{}:".format(3) + str(responses3[idx]) + "; "
    answers += "{}:".format(4) + str(responses4[idx]) + "; "

    prompt = "Consider the Question: {}\nAnswers: {}\nWhich one is the best answer?".format(questions[idx], answers)
    prompts.append(prompt)

## visualize some of the data
for i in range(5):
    print("Prompt: {}".format(prompts[i]))

### Make prediction ###

completions = []
df_pred = pd.DataFrame()
df_pred['idx'] = ''
df_pred['prompt'] = ''
df_pred['completion'] = ''
df_predRow = 0

## API key
client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = ""
)

for idx in range(len(prompts)):

    completion = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompts[idx],
        max_tokens=30,
        temperature = 0.5
    )
    response = completion.choices[0].text.strip()


    completions.append(response)
    df_pred.loc[df_predRow, 'idx'] = idx
    df_pred.loc[df_predRow, 'prompt'] = prompts[idx]
    df_pred.loc[df_predRow, 'completion'] = completions[idx]

    df_predRow += 1

df_pred.to_excel('./pref_pred.xlsx', sheet_name='Sheet1', index=False)




