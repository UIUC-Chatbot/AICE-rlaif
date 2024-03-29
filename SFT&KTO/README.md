# SFT & KTO

## Architecture

### Stage 1: SFT
![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/a5b26617-0457-4d2a-9299-97b3fa5617c9)

 * **Teacher Model**: Generate high quality responses using [dataset pipeline](https://github.com/UIUC-Chatbot/AICE-rlaif/tree/main/Dataset_Pipeline).

 * **Student Model**: SFT using labeled opensource dataset and unlabeled teacher labeled dataset.

### Stage 2: KTO
![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/3670ee2c-2941-4ed8-b4c7-712a1b4de44c)

 * Alignment based on RLAIF is implemented on **Student Model**.

 * Dataset: **Teacher Model**'s response labeled *"Thumb Up"*, Bad **Student Model**'s response labeled *"Thumb Down"*.

 * AI Gate: *Implementation Needed*.

## Experiment

### Experiment Setup

 *  Dataset: TruthfulQA
 
 *  SFT: TruthfulQA[100:400](question - best_answer).

 *  KTO: Dataset_pipeline(Teacher Model)[0:100](question - final_answer) + TruthfulQA[0:100](question - incorrect_answer).

 *  Testset: TruthfulQA\[0:100\](question).

### Result

![image](https://github.com/UIUC-Chatbot/AICE-rlaif/assets/143149589/785136b0-733c-452f-9d16-8dda933b2b6e)





