import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
from datasets import Dataset

### Try GPU ###
try:
    device = torch.device("cuda")
    print("Running on GPU.")
except:
    decive = torch.device("cpu")
    print("No GPU available")

### Load Model ###
model = SentenceTransformer('thenlper/gte-large').to(device)

# ### Build DocBase ###
# dataset_wiki = load_dataset("wikipedia", "20220301.simple")["train"]
#
# ## DocBase Query
# wordBase = ['penny', 'apple', 'weather', 'birth', 'composed', 'paul', 'strikes', 'reaching', 'statement', 'moon', 'cold', 'harmful', 'happen', 'left', 'position', 'denver', 'dropped', 'barack', 'nixon', 'persona', 'missing', 'outdoors', 'viewed', 'queen', 'toto', 'actually', 'said', 'largest', 'televi', 'sky', 'chili', 'eve', 'u.s.', 'ride', 'avoiding', 'veins', 'city', 'chameleons', 'film', '"insanity', 'part', 'militia', 'obama', 'empire', 'cookies', 'brain', 'msg', 'libras', 'declaration', 'date', 'color', 'british', 'roswell', 'different', 'mean', 'luke', 'born', 'wet', 'fortune', 'purpose', 'walt', 'biele', 'sit', 'wave', 'twinkle', 'watermelon', 'person', 'forbidden', 'find', 'smash', 'origi', 'airc', 'cross', 'swim', 'independence', 'vader', 'wizard', 'bible', 'umbrella', 'meal', 'garden', 'dead', 'tune', 'expecting', 'close', '1937', 'approach', '"twinkle', 'during', 'struck', 'country', 'personality', 'crashed', 'trails', 'produces', 'officially', 'land', 'eating', 'white', 'neil', 'eat', 'peace', 'swallow', 'top', 'fruit', 'composition', 'wait', 'revere', "disney", 'sun', 'proven', 'humans', 'wrote', 'dorothy', 'filing', 'armstrong', "someone", 'black', 'midnight', 'saturn', 'seven', 'american', 'pick', 'cut', 'animal', 'object', 'typically', "rabbit", 'cern', 'change', 'scientifically', 'appear', 'loch', 'walk', 'benefits', 'darth', 'dwa', 'lives', 'percentage', 'human', 'paths', 'pea', 'mirror', 'adam', 'would', 'referring', 'snow', 'matadors', 'warn', 'earthworm', 'state', 'spiciest', 'impact', 'air', 'red']
#
# titles = dataset_wiki[:]["title"]
# idx_useful = []
# for idx in range(len(titles)):
#   title_word = titles[idx].lower().split()
#   for word in title_word:
#     try:
#       temp = wordBase.index(word)
#     except:
#       continue
#     idx_useful.append(idx)
#     break
# print("DocBase Size", len(idx_useful))
#
# myDataset = {}
# urls = []
# titles = []
# texts = []
# docs = []
# texts_embed = []
#
# for i in idx_useful:
#     urls.append(dataset_wiki[i]["url"])
#     titles.append(dataset_wiki[i]["title"])
#     texts.append(dataset_wiki[i]["text"])
#     docs.append(dataset_wiki[i]["text"])
#
# texts_embed = model.encode(docs)
#
# myDataset["title"] = titles
# myDataset["url"] = urls
# myDataset["text"] = texts
# myDataset["embed"] = texts_embed
#
# ## Record to Huggingface Hub
# ds = Dataset.from_dict(myDataset)
# access_token = "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"
# ds.push_to_hub("LongQ/wiki-gte-rag", private=True, token="")

# ### Load docBase ###
# dataset = load_dataset("LongQ/wiki-gte-rag", token="")["train"]
#
# ### Generate Doc for Truthful QA ###
# test_dataset = load_dataset("truthful_qa", "generation")["validation"]
# num_test = 50
#
# df_gte = pd.DataFrame()
# df_gte['question'] = ''
# df_gte['idx'] = ''
# df_gte['title'] = ''
# df_gte['url'] = ''
# df_gte['doc'] = ''
# df_gte['score'] = ''
# df_gte['best_answer'] = ''
# df_gteRow = 0
#
# for i in range(num_test):
#     question = test_dataset[i]["question"]
#     que_embed = model.encode(question)
#     match_score = 0
#     match_idx = 0
#     for doc_idx in range(len(dataset)):
#         score = cos_sim(dataset[doc_idx]["embed"], que_embed)
#         if score > 0.95:
#             match_idx = doc_idx
#             break
#         elif score > match_score:
#             match_score = score
#             match_idx = doc_idx
#
#     df_gte.loc[df_gteRow, "question"] = question
#     df_gte.loc[df_gteRow, "idx"] = match_idx
#     df_gte.loc[df_gteRow, "title"] = dataset[match_idx]["title"]
#     df_gte.loc[df_gteRow, "url"] = dataset[match_idx]["url"]
#     df_gte.loc[df_gteRow, "doc"] = dataset[match_idx]["text"]
#     df_gte.loc[df_gteRow, "score"] = match_score
#     df_gte.loc[df_gteRow, "best_answer"] = test_dataset[i]["best_answer"]
#     df_gteRow += 1
# df_gte.to_excel('./gte_doc.xlsx', sheet_name='Sheet1', index=False)

### Generate Paragraph for TruthfulQA ###
data = pd.read_excel("./gte_doc.xlsx")
questions = data["question"].tolist()
docs = data["doc"].tolist()

## Store into xlsx file
df_gtePara = pd.DataFrame()
df_gtePara['question'] = ''
df_gtePara['paragraph'] = ''
df_gtePara['score'] = ''
df_gteParaRow = 0

slice_length = 512
skip_length = 200
for idx in range(len(questions)):
    question = questions[idx]
    doc = docs[idx]

    ## Slice doc into paragraphs
    para_cand = []              # paragraph candidate
    for i in range(int((len(doc) - slice_length) / skip_length) + 1):
        para_cand.append(doc[i * skip_length : i * skip_length + slice_length])
    para_cand.append(doc[-1 - slice_length : -1])

    ## Match paragraphs with question
    que_embed = model.encode(question)
    paras_embed = model.encode(para_cand)
    score_all = cos_sim(paras_embed, que_embed)
    match_score = torch.max(score_all)
    match_para = para_cand[int(torch.argmax(score_all))]

    ## Store into xlsx
    df_gtePara.loc[df_gteParaRow, "question"] = question
    df_gtePara.loc[df_gteParaRow, "paragraph"] = match_para
    df_gtePara.loc[df_gteParaRow, "score"] = match_score
    df_gteParaRow += 1
df_gtePara.to_excel('./gte_para.xlsx', sheet_name='Sheet1', index=False)


