import openai
import pandas as pd
from exa_py import Exa
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from tqdm import tqdm
from datasets import Dataset

class ds_pipe:
    def __init__(self, questions: list, rag_mt: str):

        ## store new dataset
        self.df = pd.DataFrame()
        self.df['question'] = ''
        self.df['rag_doc1'] = ''
        self.df['rag_doc2'] = ''
        self.df['rag_doc3'] = ''
        self.df['answer_org'] = ''
        self.df['critique'] = ''
        self.df['answer_final'] = ''

        ## store info
        self.questions = questions
        self.rag_mt = rag_mt            # rag method
        self.rag_docs = []              # list of list of rag docs
        self.df = pd.DataFrame()        # store new dataset
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")            # check gpu

    def rag(self, exa_key: str):

        print("Retrieving Documents...")

        ## store rag
        df_rag = pd.DataFrame()
        df_rag['question'] = ''
        df_rag['rag_doc1'] = ''
        df_rag['rag_doc2'] = ''
        df_rag['rag_doc3'] = ''

        for idx in range(len(self.questions)):
            df_rag.loc[idx, "question"] = self.questions[idx]

        for idx in tqdm(range(len(self.questions))):

            sample_docs = []
            embedder = SentenceTransformer('thenlper/gte-large').to(self.device)
            que_embed = embedder.encode(self.questions[idx])

            if self.rag_mt == "exa":
                exa = Exa(exa_key)
                docs = exa.search_and_contents(self.questions[idx], num_results=3, use_autoprompt=True)

                doc_idx = 0
                for doc in docs.results:
                    sample_docs.append(doc.text)
                    doc_idx += 1

            elif self.rag_mt == "gte":

                ## Load docBase
                docBase = load_dataset("LongQ/wiki-gte-rag", token="")["train"]

                match_docs = {}

                for doc_idx in range(len(docBase)):
                    score = cos_sim(docBase[doc_idx]["embed"], que_embed)
                    match_docs[doc_idx] = score

                ## collect top 3 docs
                match_docs = sorted(match_docs.items(), key=lambda x: x[1], reverse=True)[0:3]
                for i in range(3):
                    sample_docs.append(docBase[match_docs[i][0]]["text"])

            else:
                print("Error caught: Currently we only support [exa] and [gte].")
                return

            sample_para = []
            ## match para for each doc
            for doc_idx in range(len(sample_docs)):
                doc = sample_docs[doc_idx]

                ## Slice doc into paragraphs
                slice_length = 512
                skip_length = 200
                para_cand = []              # paragraph candidate
                for i in range(int((len(doc) - slice_length) / skip_length) + 1):
                    para_cand.append(doc[i * skip_length: i * skip_length + slice_length])
                para_cand.append(doc[-1 - slice_length: -1])

                ## Match paragraphs with question
                paras_embed = embedder.encode(para_cand)
                score_all = cos_sim(paras_embed, que_embed)
                match_para = para_cand[int(torch.argmax(score_all))]

                sample_para.append(match_para)
                self.df.loc[idx, "rag_doc{}".format(doc_idx)] = match_para
                df_rag.loc[idx, "rag_doc{}".format(doc_idx)] = match_para

            self.rag_docs.append(sample_para)

        df_rag.to_excel("./rag_docs.xlsx", sheet_name='Sheet1', index=False)

    def generate_response(self, anyscale_key: str):

        print("Generating Answers...")

        ## API key
        client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=anyscale_key
        )

        ## Combine questions and paragraphs
        for idx in tqdm(range(len(self.questions))):
            prompt = ""
            for i in range(len(self.rag_docs[idx])):
                prompt += "\"" + self.rag_docs[idx][i] + "\"\n"
            prompt += "\nBased on these paragraphs and true fact, " + self.questions[idx]

            ## Generate base response
            completion = client.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                prompt=prompt,
                max_tokens=30,
                temperature=0.5
            )
            response_base = completion.choices[0].text.strip()
            self.df.loc[idx, "answer_org"] = response_base

            ## Critique
            prompt = ""
            for i in range(len(self.rag_docs[idx])):
                prompt += "\"" + self.rag_docs[idx][i] + "\"\n"
            prompt = "Consider the Question: {}\nanswer: {}\nPlease evaluate and improve this answer in a EVALUATION and IMPROVE manner.\n\n###".format(self.questions[idx], response_base)

            completion = client.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf:qi:2HP7r23",
                prompt=prompt,
                max_tokens=200,
                temperature = 0.5
            )
            response_critique = completion.choices[0].text.strip()
            self.df.loc[idx, "critique"] = response_critique

            ## Final answer
            prompt = ""
            for i in range(len(self.rag_docs[idx])):
                prompt += "\"" + self.rag_docs[idx][i] + "\"\n"
            prompt += "\nBased on these paragraphs and true fact, " + self.questions[idx]
            prompt += "\nConsider your previous answer: {}\n".format(response_base) \
                     + " and Evaluation: {}\n".format(response_critique) \
                     + "Please give your final answer to question: {}".format(self.questions[idx])

            completion = client.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                prompt=prompt,
                max_tokens=30,
                temperature=0.5
            )
            response_final = completion.choices[0].text.strip()
            self.df.loc[idx, "answer_final"] = response_final

    def build_ds(self, output_dir: str):
        print("Storing Dataset...")
        for idx in range(len(self.questions)):
            self.df.loc[idx, "question"] = self.questions[idx]
        self.df.to_excel("./" + output_dir + ".xlsx", sheet_name='Sheet1', index=False)



### Call pipeline ###
ds_len = 100

## OpenHermes-2.5
oh_questions = []
oh_dataset = load_dataset("teknium/OpenHermes-2.5")["train"][0:ds_len]["conversations"]
for idx in range(len(oh_dataset)):
    sample = oh_dataset[idx]
    for j in range(len(sample)):
        if sample[j]["from"] == "human":
            oh_questions.append(sample[j]["value"])
            continue

## TruthfulQA
tf_questions = load_dataset("truthful_qa", "generation")["validation"][0:ds_len]["question"]

pipe = ds_pipe(tf_questions, "exa")
pipe.rag("")
pipe.generate_response("")
pipe.build_ds("TruthfulQA_dataset")

### Upload to huggingface
myDataset = {}

data = pd.read_excel("./TruthfulQA_dataset.xlsx")
questions = data["question"].tolist()
rag_doc0 = data["rag_doc0"].tolist()
rag_doc1 = data["rag_doc1"].tolist()
rag_doc2 = data["rag_doc2"].tolist()
answers_org = data["answer_org"].tolist()
critiques = data["critique"].tolist()
answers_final = data["answer_final"].tolist()

## combine docs
rag_docs = []
for i in range(len(questions)):
    sample_docs = []
    if type(rag_doc0[i]) == str and len(rag_doc0[i]) > 1:
        sample_docs.append(rag_doc0[i])
    if type(rag_doc1[i]) == str and len(rag_doc1[i]) > 1:
        sample_docs.append(rag_doc1[i])
    if type(rag_doc2[i]) == str and len(rag_doc2[i]) > 1:
        sample_docs.append(rag_doc2[i])
    rag_docs.append(sample_docs)

## load to dict
myDataset["question"] = questions
myDataset["rag_docs"] = rag_docs
for i in range(len(answers_org)):
    if type(answers_org[i]) == float:
        answers_org[i] = "nan"
myDataset["org_ans"] = answers_org
myDataset["critique"] = critiques
myDataset["final_answer"] = answers_final

ds = Dataset.from_dict(myDataset)
ds.push_to_hub("LongQ/TruthfulQA_critique_small", private=True, token="")















