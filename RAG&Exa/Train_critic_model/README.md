# utils
This document includes important functions.

# train a model
By using [self_critique_RAG.py], we can train a model with our own dataset(here I used self-rag dataset) and push the fine-tuned model to hugging face.\
Here I fine-tuned ["teknium/OpenHermes-2.5-Mistral-7B"] model.

# example
By using [example.py], I used my fine-tuned model to generate answers based on retrieved messages.\
I embedded the self-rag paper and asked a specific question about this paper.

# notice
Everything needs to be refined!!!\
I need to adjust the datasets. And I'll re-orginize my code so that you guys can use it directly later ^_^
