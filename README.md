# AICE-rlaif

## Technical description:

We aim to research, design, and implement a conversational AI that can rapidly become an “expert” on a user-defined topic.

Our team at NCSA has been developing an AI chatbot platform to create “teaching assistants” trained on course materials [1]. Our prototype is capable of rapidly ingesting instructor-supplied course materials (textbook, instructor notes, lecture slides, lecture videos) and using this information: on-to-ground GPT-4 model in order to respond to student questions related to the course. In the process of implementing this framework, we identified opportunities for such chatbots to become very effective in answering questions within a narrowly defined area as well as challenges in adapting large language models (LLMs) for such tasks. With this proposal, we are looking to address some of these challenges and develop and demonstrate a framework capable of quickly becoming an “expert” on a user-defined topic. 

The proposed system will operate as follows:

1. A user specifies the source or subject, e.g., a biology textbook for the class they are taking, a series of historical documents for the research paper they are writing, or a literature piece they are reading for the book club, to name a few.
2. LLM(s) is “trained” on these sources, relevant additional materials, and their derivatives. To include deeper levels of primary materials for each topic, we will include in the training corpora all references in the citation tree of each source. Additionally, separate LLMs will use the primary material to generate task-specific versions of the data, e.g., Q&A pairs, “explain like I’m 5” answers, summaries, and more.
3. The user interacts with the newly tuned LLM via a chatbot interface, either text-based or via voice.

## Key innovations will include:

1. Scalable Oversight: We aim to refine generated responses using a factual consistency model. This model will evaluate whether the answer is backed by a corpus of verified information sources like textbooks and scientific databases [2]. Non-factual responses are self-corrected using prompt engineering to force models to s:ck to the facts.
  Appendix A details prompts for (1) question generation, (2) answer generation, and (3) answer scoring with and without ground truth answers. Appendix B shows the Evaluator application that will enable our work on large-scale corpora.
2. Novel factuality loss in RL with AI Feedback. We will introduce a novel training penalty, beyond cross entropy, termed factuality loss, a method of retrieval-augmented RL with AI feedback. Given a document, textbook, or conference paper, we preprocess it by generating Q&A pairs (as in contribution 1 above). During training, retrieved documents are included in the prompt and the generated answers are evaluated by our factual consistency model that directly produces a final reward value. As we’ve done before, we will use TRLX for distributed RLHF/RLAIF training.
This additional factuality loss builds on the fundamental successes of Reinforcement Learning from Human Feedback [8] (RLHF) and Anthropic’s Constitutional AI [9], in a deep research effort to teach the model that hallucination is never acceptable.
3. Supervise the reasoning process in addition to outcomes. In addition to (1) we propose decomposing the questions into PageRank style considerations such as “Is the author an expert in this field? Do they have prior work on this topic? “, etc. We assert that by forcing the model to think through the process of research, in an extended ‘chain of thought’ loop, it will be capable of both world-class question answering and even the creation of new knowledge. The models will have access to tools such as relevant scientific databases, WolframAlpha, and more [3-6], enabling to verification of hypotheses by direct examination of existing data and even proposing new experiments to fill gaps in the existing literature.
