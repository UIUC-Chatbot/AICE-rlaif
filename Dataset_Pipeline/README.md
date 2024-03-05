# Dataset_Pipeline

## Architecture
![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/02e60dc0-26ef-43c3-bdfb-9a65b5ddee24)

 *  Stage 1 - **RAG**: two strategies available [EXA](https://exa.ai/), [GTE DocBase](https://huggingface.co/datasets/LongQ/wiki-gte-rag)

 *  Stage 2 - Initial Response: Prompt - Doc + Question, Completion - Answer.

 *  Stage 3 - Self-critique: Finetuned **Improver** Model.

 *  Stage 4 - Final Response: Prompt - Doc + Question + Initial Response + Critique, Completion - Answer.

## Usage

### Class Overview
```
class ds_pipe:
    def __init__(self, questions: list, rag_mt: str):
    def rag(self, exa_key: str):
    def generate_response(self, anyscale_key: str):
    def build_ds(self, output_dir: str):
```
