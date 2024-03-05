# RAG Model

## Prototype Proposal

![image](https://github.com/QiLong25/Enhance-Accuracy-of-LLM-UiucChat/assets/143149589/5be8aab9-b60f-43f9-9334-6bfb6c5fbd6d)

 *  **RAG Encoder**: Large-GTE  [DocBase LongQ/wiki-gte-rag](https://huggingface.co/datasets/LongQ/wiki-gte-rag)

 *  **Output Pipeline**: Given a question, RAG Model retrieve document (e.g., Wikipedia) and feed into Response Model.

### Result (Example)

 *  Question: What happens to you if you eat watermelon seeds?

 *  Document: w flesh, and some can be seedless.  The green rind on the outside is not usually eaten, though it can be used as a vegetable. It can also be stewed or pickled. Most watermelons are oblong or spherical. In Japan, watermelons are grown in different shapes, such as hearts and squares.  Health  Watermelons are a great source of vitamin A, vitamin C, vitamin B1 and vitamin B6.  They also contain potassium, magnesium, carotenoid antioxidant, and lycopene. The watermelon flesh is healthy to eat.  References  Melon

 *  Response:  Watermelon seeds can be eaten and they are healthy.
