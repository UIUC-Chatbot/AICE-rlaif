from datasets import load_dataset
import random
import pandas as pd
import openai
import json

# ### Prepare dataset ###
#
# df_impr = pd.DataFrame()
# df_impr['question'] = ''
# df_impr['best'] = ''
# df_impr['answer2'] = ''
# df_impr['prompt'] = ''
# df_impr['completion'] = ''
# df_imprRow = 0
#
# dataset = load_dataset("truthful_qa", "generation")
# dataset = dataset["validation"]
# print(len(dataset))
# print(dataset[0])
# print("Dataset Loading finished.\n")
#
# questions = []
# gold_best = []    # single optimal gold answer
# gold_correct = []     # list of lists
# gold_incorrect = []
#
# prompts = []             # for finetuning
# labels = []
#
# sample_num = 150
# sample_start = 200
#
# for idx in range(sample_start, sample_start + sample_num):
#     questions.append(dataset[idx]["question"])
#     gold_best.append(dataset[idx]["best_answer"])
#     gold_correct.append(dataset[idx]["correct_answers"])
#     gold_incorrect.append(dataset[idx]["incorrect_answers"])
#
# for idx in range(len(questions)):
#
#     if random.randint(1, 2) == 1:           # choose correct answer as target
#         for corr in gold_correct[idx]:
#             if gold_best[idx] == corr:
#                 continue
#             prompt = "Consider the Question: {}\nAnswer1: {}\nAnswer2: {}\nPlease give reasons why Answer2 is worse than Answer1.".format(questions[idx], gold_best[idx], corr)
#             df_impr.loc[df_imprRow, "question"] = questions[idx]
#             df_impr.loc[df_imprRow, "best"] = gold_best[idx]
#             df_impr.loc[df_imprRow, "answer2"] = corr
#             df_impr.loc[df_imprRow, "prompt"] = prompt
#             df_imprRow += 1
#         for incorr in gold_incorrect[idx]:
#             prompt = "Consider the Question: {}\nAnswer1: {}\nAnswer2: {}\nPlease give reasons why Answer2 is incorrect compared to Answer1".format(questions[idx], gold_best[idx], incorr)
#             df_impr.loc[df_imprRow, "question"] = questions[idx]
#             df_impr.loc[df_imprRow, "best"] = gold_best[idx]
#             df_impr.loc[df_imprRow, "answer2"] = incorr
#             df_impr.loc[df_imprRow, "prompt"] = prompt
#             df_imprRow += 1

# ## save to excel
# df_impr.to_excel('./impr/impr_train.xlsx', sheet_name='Sheet1', index=False)

# ### Generate Completion ###
# data = pd.read_excel("./impr/impr_train.xlsx")
# questions = data["question"].tolist()
# bests = data["best"].tolist()
# answer2s = data["answer2"].tolist()
# prompts = data["prompt"].tolist()
# # sample_num = 10
# # prompts = prompts[0:sample_num]
# # print(prompts)
#
# df_impr2 = pd.DataFrame()
# df_impr2['question'] = ''
# df_impr2['best'] = ''
# df_impr2['answer2'] = ''
# df_impr2['prompt'] = ''
# df_impr2['completion'] = ''
# df_impr2Row = 0
#
# client = openai.OpenAI(
#     base_url = "https://api.endpoints.anyscale.com/v1",
#     api_key = "esecret_vivx2qwtxplg4fswxljg25wl38"
# )
#
# for idx in range(len(prompts)):
#     if idx % 100 == 0:
#         print("Generating completion for prompt {}".format(idx))
#     completion = client.completions.create(
#         model="meta-llama/Llama-2-7b-chat-hf",
#         prompt=prompts[idx],
#         max_tokens=300,
#         temperature=0.5
#     )
#     df_impr2.loc[df_impr2Row, "question"] = questions[idx]
#     df_impr2.loc[df_impr2Row, "best"] = bests[idx]
#     df_impr2.loc[df_impr2Row, "answer2"] = answer2s[idx]
#     df_impr2.loc[df_impr2Row, "prompt"] = prompts[idx]
#     df_impr2.loc[df_impr2Row, "completion"] = completion.choices[0].text.strip()
#     df_impr2Row += 1
#
# df_impr2.to_excel('./impr/impr_train3.xlsx', sheet_name='Sheet1', index=False)

# ### Clean dataset ###
# data = pd.read_excel("./impr/impr_train3.xlsx")
# questions = data["question"].tolist()
# bests = data["best"].tolist()
# answer2s = data["answer2"].tolist()
# completions = data["clean"].tolist()
#
# df_impr_clean = pd.DataFrame()
# df_impr_clean['prompt'] = ''
# df_impr_clean['completion'] = ''
# df_imprCRow = 0
#
# ## clean completions
# idx = 0
# while idx < len(completions):
#     if len(completions[idx]) <= 5:
#         del questions[idx]
#         del bests[idx]
#         del answer2s[idx]
#         del completions[idx]
#         idx -= 1
#     elif completions[idx][0] == "." and completions[idx][1] == "\n":
#         completions[idx] = completions[idx][2:]
#     idx += 1
#
# ## generate prompts and completions
# for idx in range(len(questions)):
#     prompt = "Consider the Question: {}\nanswer: {}\nPlease evaluate and improve this answer in a EVALUATION and IMPROVE manner.\n\n###".format(questions[idx], answer2s[idx])
#     completion = "###\n\nEVALUATION:\n{}\n\nIMPROVE:\nConsider all the above, a better answer is: {}\n\n###END".format(completions[idx], bests[idx])
#     df_impr_clean.loc[df_imprCRow, "prompt"] = prompt
#     df_impr_clean.loc[df_imprCRow, "completion"] = completion
#     df_imprCRow += 1
# df_impr_clean.to_excel('./impr/impr_train_clean.xlsx', sheet_name='Sheet1', index=False)

# ### Build training set ###
# data = pd.read_excel("./impr/impr_train_clean.xlsx")
# user_content = data["prompt"].tolist()
# assistant_content = data["completion"].tolist()
#
# with open("impr_train.json", "w", encoding='utf-8') as f:
#
#     for idx in range(len(user_content)):
#         message = []
#         message_dict = {}
#
#         user_dict = {}
#         user_dict["role"] = "user"
#         user_dict["content"] = user_content[idx]
#         message.append(user_dict)
#
#         assistant_dict = {}
#         assistant_dict["role"] = "assistant"
#         assistant_dict["content"] = assistant_content[idx]
#         message.append(assistant_dict)
#
#         message_dict["messages"] = message
#
#         f.write(json.dumps(message_dict))
#         f.write("\n")
#
# ### Validate dataset ###
# DATA_PATH = "./impr_train.json"
#
# # Load the dataset
# with open(DATA_PATH, 'r', encoding='utf-8') as f:
#     items = [json.loads(line) for line in f]
#
# class DataFormatError(Exception):
#     pass
#
# def check_data_for_format_errors(items: list):
#
#     for line_num, batch in enumerate(items):
#         prefix = f"Error in line #{line_num + 1}: "
#         if not isinstance(batch, dict):
#             raise DataFormatError(
#                 f"{prefix}Each line in the provided data should be a dictionary"
#             )
#
#         if "messages" not in batch:
#             raise DataFormatError(
#                 f"{prefix}Each line in the provided data should have a 'messages' key"
#             )
#
#         if not isinstance(batch["messages"], list):
#             raise DataFormatError(
#                 f"{prefix}Each line in the provided data should have a 'messages' key with a list of messages"
#             )
#
#         messages = batch["messages"]
#         if not any(message.get("role", None) == "assistant" for message in messages):
#             raise DataFormatError(
#                 f"{prefix}Each message list should have at least one message with role 'assistant'"
#             )
#
#         for message_num, message in enumerate(messages):
#             prefix = f"Error in line #{line_num + 1}, message #{message_num + 1}: "
#             if "role" not in message or "content" not in message:
#                 raise DataFormatError(
#                     f"{prefix}Each message should have a 'role' and 'content' key"
#                 )
#
#             if any(k not in ("role", "content", "name") for k in message):
#                 raise DataFormatError(
#                     f"{prefix}Each message should only have 'role', 'content', and 'name' keys, any other key is not allowed"
#                 )
#
#             if message.get("role", None) not in ("system", "user", "assistant"):
#                 raise DataFormatError(
#                     f"{prefix}Each message should have a valid role (system, user, or assistant)"
#                 )
#
#
# try:
#     check_data_for_format_errors(items)
#     print("Data format is valid!")
# except DataFormatError as e:
#     print("Data format is NOT valid!")
#     print(e)

### Finetune ###

## API key
client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = "esecret_vivx2qwtxplg4fswxljg25wl38"
)

# ## upload train set
# file_name = "./impr_train.json"
# file = client.files.create(
#   file=open(file_name, "rb"),
#   purpose="fine-tune",
# )

## finetune
client.fine_tuning.jobs.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    training_file="file_4icackpacjc9pdm7n9nwlnjmw2",
    hyperparameters={"n_epochs":20, },
)
