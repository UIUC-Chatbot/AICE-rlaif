# Dataset_Pipeline

## Architecture
![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/02e60dc0-26ef-43c3-bdfb-9a65b5ddee24)

 *  Stage 1 - **RAG**: two strategies available [EXA](https://exa.ai/), [GTE DocBase](https://huggingface.co/datasets/LongQ/wiki-gte-rag)

 *  Stage 2 - Initial Response: Prompt - Doc + Question, Completion - Answer.

 *  Stage 3 - Self-critique: Finetuned **Improver** Model.

 *  Stage 4 - Final Response: Prompt - Doc + Question + Initial Response + Critique, Completion - Answer.

## Usage

### Class Overview
```python
class ds_pipe:
    def __init__(self, questions: list, rag_mt: str):
    def rag(self, exa_key: str):
    def generate_response(self, anyscale_key: str):
    def build_ds(self, output_dir: str):
```

### Call Pipeline
```python
pipe = ds_pipe(questions, <RAG_METHOD>)
pipe.rag("<EXA_API_KEY>")
pipe.generate_response("<ANYSCALE_API_KEY>")
pipe.build_ds("<OUTPUT_EXCEL_FILE_NAME>")
```

 *  questions: a list of questions.

 *  <RAG_METHOD>: "exa" or "gte".

 *  output file: to current directory as default.

### Example Call
```python
## TruthfulQA
ds_len = 100
tf_questions = load_dataset("truthful_qa", "generation")["validation"][0:ds_len]["question"]

pipe = ds_pipe(tf_questions, "exa")
pipe.rag("xxxxxxxxxxx")
pipe.generate_response("xxxxxxxxxxxxx")
pipe.build_ds("TruthfulQA_dataset")
```

## Output Example (Truthful QA)
![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/c202db62-52c0-490c-8706-9177a5d85166)


