from datasets import load_dataset
import random
import pandas as pd
import openai
import json

### Prepare dataset ###

df_pref = pd.DataFrame()
df_pref['idx'] = ''
df_pref['prompt'] = ''
df_pref['label'] = ''
df_prefRow = 0

dataset = load_dataset("truthful_qa", "generation")
dataset = dataset["validation"]
print(len(dataset))
print(dataset[0])
print("Dataset Loading finished.\n")

questions = []
gold_best = []    # single optimal gold answer
gold_correct = []     # list of lists
gold_incorrect = []

prompts = []             # for finetuning
labels = []

sample_num = 100
sample_start = 50

for idx in range(sample_start, sample_start + sample_num):
    questions.append(dataset[idx]["question"])
    gold_best.append(dataset[idx]["best_answer"])
    gold_correct.append(dataset[idx]["correct_answers"])
    gold_incorrect.append(dataset[idx]["incorrect_answers"])

for idx in range(len(questions)):
    candidate_dict = {}             # text : label
    candidate_dict[gold_best[idx]] = "best"
    for i in range(len(gold_correct[idx])):
        candidate_dict[gold_correct[idx][i]] = "correct"
    for j in range(len(gold_incorrect[idx])):
        candidate_dict[gold_incorrect[idx][j]] = "incorrect"
    order = []
    for i in range(len(candidate_dict)):
        order.append(i+1)
    random.shuffle(order)          # mix up good and bad answers
    candidate = list(candidate_dict.keys())
    answers = ""
    label = "Good ones are "
    display_idx = 1
    for j in order:
        answers += "{}:".format(display_idx) + candidate[j-1] + "; "
        if candidate_dict[candidate[j-1]] != "incorrect":
            label += "{}, ".format(display_idx)
        display_idx += 1
    label += "the rest are bad ones."
    prompt = "Consider the Question: {}\nAnswers: {}\nWhich are good ones and which are bad ones?".format(questions[idx], answers)
    prompts.append(prompt)
    labels.append(label)
    df_pref.loc[df_prefRow, 'idx'] = idx
    df_pref.loc[df_prefRow, 'prompt'] = prompt
    df_pref.loc[df_prefRow, 'label'] = label
    df_prefRow += 1

for i in range(5):
    print("Prompt: {}".format(prompts[i]))
    print("Label: {}".format(labels[i]))

## save to excel
df_pref.to_excel('./pref_train.xlsx', sheet_name='Sheet1', index=False)

### Build training set ###
data = pd.read_excel("./pref_train.xlsx")
user_content = data["prompt"].tolist()
assistant_content = data["label"].tolist()

with open("pref_train.json", "w", encoding='utf-8') as f:

    for idx in range(len(user_content)):
        message = []
        message_dict = {}

        user_dict = {}
        user_dict["role"] = "user"
        user_dict["content"] = user_content[idx]
        message.append(user_dict)

        assistant_dict = {}
        assistant_dict["role"] = "assistant"
        assistant_dict["content"] = assistant_content[idx]
        message.append(assistant_dict)

        message_dict["messages"] = message

        f.write(json.dumps(message_dict))
        f.write("\n")

### Validate dataset ###
DATA_PATH = "./pref_train.json"

# Load the dataset
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

class DataFormatError(Exception):
    pass

def check_data_for_format_errors(items: list):

    for line_num, batch in enumerate(items):
        prefix = f"Error in line #{line_num + 1}: "
        if not isinstance(batch, dict):
            raise DataFormatError(
                f"{prefix}Each line in the provided data should be a dictionary"
            )

        if "messages" not in batch:
            raise DataFormatError(
                f"{prefix}Each line in the provided data should have a 'messages' key"
            )

        if not isinstance(batch["messages"], list):
            raise DataFormatError(
                f"{prefix}Each line in the provided data should have a 'messages' key with a list of messages"
            )

        messages = batch["messages"]
        if not any(message.get("role", None) == "assistant" for message in messages):
            raise DataFormatError(
                f"{prefix}Each message list should have at least one message with role 'assistant'"
            )

        for message_num, message in enumerate(messages):
            prefix = f"Error in line #{line_num + 1}, message #{message_num + 1}: "
            if "role" not in message or "content" not in message:
                raise DataFormatError(
                    f"{prefix}Each message should have a 'role' and 'content' key"
                )

            if any(k not in ("role", "content", "name") for k in message):
                raise DataFormatError(
                    f"{prefix}Each message should only have 'role', 'content', and 'name' keys, any other key is not allowed"
                )

            if message.get("role", None) not in ("system", "user", "assistant"):
                raise DataFormatError(
                    f"{prefix}Each message should have a valid role (system, user, or assistant)"
                )


try:
    check_data_for_format_errors(items)
    print("Data format is valid!")
except DataFormatError as e:
    print("Data format is NOT valid!")
    print(e)

### Finetune ###

## API key
client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = ""
)

## upload train set
file_name = "./pref_train.json"
file = client.files.create(
  file=open(file_name, "rb"),
  purpose="fine-tune",
)

## finetune
client.fine_tuning.jobs.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    training_file="file_gl8gdjh82ubqza1yz1wfk52pyt",
    hyperparameters={"n_epochs":3, },
)

