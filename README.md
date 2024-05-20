# Large Language Models Interview Questions
This repository contains interview questions about Large Language Models (LLMs). I'm basing it off of [Mastering LLM](https://www.masteringllm.com/course/llm-interview-questions-and-answers), credit to them for getting this project started, however, my contribution will be the answers.

7 questions answered out of ??.

## Table of Contents
In the original material, the questions are divided into the following chapters:
1. Road map
2. Prompt engineering & basics of LLM
3. Retrieval augmented generation (RAG)
4. Chunking strategies
5. Embedding Models
6. Internal working of vector DB
7. Advanced search algorithms
8. Language models internal working
9. Supervised fine-tuning of LLM
10. Preference Alignment (RLHF/DPO)
11. Evaluation of LLM system
12. Hallucination control techniques
13. Deployment of LLM
14. Agent-based system
15. Prompt Hacking
16. Case Study & Scenario-based question

I will be loosly following the same order.

## What is the difference between Predictive/Discriminative and Generative AI Models?
Let's consider a dataset, where each data point represents a cat. Let's pass it through to each type of model, and see how they differ:
- **Predictive/Discriminative Model**: This type of model will either try to discriminate between different data point, by virtue of a decision boundary, and a mapping function, or by predicting the probability of a class label given the model's ability to learn patterns from the data.
- **Generative Model**: This type of model will try to generate new data points, by learning the underlying distribution of the data given all the data points in the dataset.

## What is a Large Language Model (LLM)?
Let's build the definition of a Large Language Model (LLM) from the ground up:
- **Language Models (LMs)**: These are probabilistic models that learn from natural language, they can range from simple bayesian models to more complex neural networks.
- **Large**: Consider all the data available, a literal dump of the internet, enough compute to process and learn from it, and a model that can handle the complexity of the data.
- **Large Language Model**: The amalgamation of the two, a model that can learn from a large amount of data, and generate text that is coherent and human-like.

Further reading:
[Common Crawl](https://commoncrawl.org/)

## How are Large Language Models trained?
Large Language Models are often trained in multiple stages, these stages are often named pretraining, fine-tuning, and alignment. 
### Pretraining: 
The purpose of this stage is to expose the model to _all of language_, in an unsupervised manner, it is often the most expensive part of training, and requires a lot of compute. Pretraining is often done on something like the [Common Crawl](https://commoncrawl.org/) dataset, processed versions of the dataset such as [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [RedPajama](https://github.com/togethercomputer/RedPajama-Data) are often used for pretraining.
To facilitate this broad type of learning, there exists multiple training tasks we can use, such as Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and more.

Mask Language Modeling is based of the [Cloze Test](https://en.wikipedia.org/wiki/Cloze_test), where we mask out a word in a sentence, and ask the model to predict it. Similar to a fill in the blank test. It differs from asking the model to predict the next word in a sentence, as it requires the model to understand the context of the sentence, and not just the sequence of words.

Next Sentence Prediction is a task where the model is given two sentences, and it has to predict if the second sentence follows the first one. As simple as it sounds, it requires the model to understand the context of the first sentence, and the relationship between the two sentences.
An excellent resource to learn more about these tasks is the [BERT paper](https://arxiv.org/abs/1810.04805).

### Fine-tuning:
This stage is much simpler than pretraining, as the model has already learned a lot about language, and now we just need to teach it about a specific task. All we need for this stage is the input data (prompts) and the labels (responses). 

### Alignment:
This stage is often the most cruical and complex stage, it requires the use of seperate reward models, the use of different learning paradigms such as Reinforcement Learning, and more. 

This stage mainly aims to align the model's predictions with the human's preferences. This stage often interweaves with the fine-tuning stage. Essential reading for this stage is the [InstructGPT paper](https://arxiv.org/pdf/2203.02155), this paper introduced the concept of Reward Learning from Human Feedback (RLHF) which uses [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347).

Other methods of Aligning the model's predictions with human preferences include:
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)

## What is a token in the context of LLMs?
Tokens are the smallest unit of text that the model can understand, they can be words, subwords, or characters.

## What are tokenizers in the context of LLMs?
Tokenizers are responsiple for converting text into tokens, they can be as simple as splitting the text by spaces, or as complex as using subword tokenization. The choice of tokenizer can have a significant impact on the model's performance, as it can affect the model's ability to understand the context of the text.

Some common tokenizers include:
- [Byte Pair Encoding (BPE)](https://aclanthology.org/P16-1162.pdf) 
- [WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [SentencePiece](https://arxiv.org/pdf/1808.06226)

Recommended reading (and watching):
- [Summary of Tokenizers by HuggingFace](https://huggingface.co/docs/transformers/en/tokenizer_summary)
- [Let's build the GPT Tokenizer by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)

## How do you estimate the cost of running a SaaS-based closed source LLM against an open-source LLM?
This is a very loaded question, but here are some resources to explore this topic further:
- [Using LLMs for Enterprise Use Cases: How Much Does It Really Cost?](https://www.titanml.co/resources/using-llms-for-enterprise-use-cases-how-much-does-it-really-cost)
- [The Challenges of Self-Hosting Large Language Models](https://www.titanml.co/resources/the-challenges-of-self-hosting-large-language-models)
- [The Case for Self-Hosting Large Language Models](https://www.titanml.co/resources/the-case-for-self-hosting-large-language-models)
- [Exploring the Differences: Self-hosted vs. API-based AI Solutions](https://www.titanml.co/resources/exploring-the-differences-self-hosted-vs-api-based-ai-solutions)

## When you run inference on an LLM, you can pick hyperparameters such as temperature, top-k, top-p, and repetition penalty. What do these hyperparameters do?
Recommended reading:
- [How to tune LLM Parameters for optimal performance](https://datasciencedojo.com/blog/tuning-optimizing-llm-parameters/)
- [OpenAI Documentation](https://platform.openai.com/docs/guides/text-generation)
- [LLM Settings by Prompt Engineering Guide](https://www.promptingguide.ai/introduction/settings)

## What are the different decoding strategies for picking output tokens?
Decoding strategies are used to pick the next token in the sequence, they can range from simple greedy decoding to more complex sampling strategies. 

Some common decoding strategies include:
- Greedy Decoding
- Beam Search

Fancy decoding strategies include Speculative Decoding which is a wild concept, it involves using a template from a weaker model to generate a response from a stronger model very quickly.

Recommended reading:
- [Text generation strategies by HuggingFace](https://huggingface.co/docs/transformers/generation_strategies)

## What are the stopping criteria for decoding in the context of LLMs?
In the decoding process, LLMs autoregressively generate text one token at a time. There are several stopping criteria that can be used to determine when to stop generating text. Some common stopping criteria include:
- Maximum Length: Stop generating text when the generated text reaches a certain length.
- End of Sentence Token: Stop generating text when the model generates an end of sentence token.



