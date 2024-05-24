# Large Language Models Interview Questions
This repository contains interview questions about Large Language Models (LLMs). 

I am using this reference: [Mastering LLM](https://www.masteringllm.com/course/llm-interview-questions-and-answers) as the base, credit to them for compiling it, however, I am taking _alot_ of liberies with editing the questions, as well as the answers, they are completely my own.

_Note:_ I am trying to keep the answers I write myself to a minumum, since I am in no way or form an authoritative source on this topic. I will be providing references to the best of my ability.

## Preamble
If you've reached the stage where you're interviewing for Generative AI roles, this will help you a lot. 

If you're just getting started, my one and only piece of advice is:

    Get comfortable reading papers, because there's so many papers. The papers never end.

## Table of Contents
In the original material, the questions are divided into the following chapters:
1. ~~Road map~~
2. ~~Prompt engineering & basics of LLM~~
3. ~~Retrieval augmented generation (RAG)~~
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
- **Predictive/Discriminative Model**: This type of model will either try to discriminate between different data points, by virtue of a decision boundary, and a mapping function, or by predicting the probability of a class label given the model's ability to learn patterns from the data.
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

## How do you estimate the cost of running an API based / closed source LLM vs. an open-source LLMs (self-hosted)?
This is a very loaded question, but here are some resources to explore this topic further:
- [Using LLMs for Enterprise Use Cases: How Much Does It Really Cost?](https://www.titanml.co/resources/using-llms-for-enterprise-use-cases-how-much-does-it-really-cost)
- [The Challenges of Self-Hosting Large Language Models](https://www.titanml.co/resources/the-challenges-of-self-hosting-large-language-models)
- [The Case for Self-Hosting Large Language Models](https://www.titanml.co/resources/the-case-for-self-hosting-large-language-models)
- [Exploring the Differences: Self-hosted vs. API-based AI Solutions](https://www.titanml.co/resources/exploring-the-differences-self-hosted-vs-api-based-ai-solutions)

## What are the different parameters that can be tuned in LLMs during inference?
Parameters include:
- Temperature
- Top P
- Max Length
- Stop Sequences
- Frequency Penalty
- Presence Penalty

Each of these parameters can be tuned to improve the performance of the model, and the quality of the generated text.

Recommended reading:
- [How to tune LLM Parameters for optimal performance](https://datasciencedojo.com/blog/tuning-optimizing-llm-parameters/)
- [OpenAI Documentation](https://platform.openai.com/docs/guides/text-generation)
- [LLM Settings by Prompt Engineering Guide](https://www.promptingguide.ai/introduction/settings)

## What are the different decoding strategies for picking output tokens?
Decoding strategies are used to pick the next token in the sequence, they can range from simple greedy decoding to more complex sampling strategies. 

Some common decoding strategies include:
- Greedy Decoding
- Beam Search

Newer decoding strategies include Speculative Decoding (assisted decoding) which is a wild concept, it involves using a candidate tokens from a smaller (thus faster) model to generate a response from a bigger model very quickly.

Recommended reading:
- [Text generation strategies by HuggingFace](https://huggingface.co/docs/transformers/generation_strategies)
- [Speculative Decoding Paper](https://arxiv.org/pdf/2211.17192.pdf)
- [A Hitchhiker’s Guide to Speculative Decoding by Team PyTorch at IBM](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

## What are the stopping criteria for decoding in the context of LLMs?
In the decoding process, LLMs autoregressively generate text one token at a time. There are several stopping criteria that can be used to determine when to stop generating text. Some common stopping criteria include:
- Maximum Length: Stop generating text when the generated text reaches a certain length.
- End of Sentence Token: Stop generating text when the model generates an end of sentence token.
- Stop Sequences: Stop generating text when the model generates a predefined stop sequence.

## What are some elements that make up a prompt?
```
A prompt contains any of the following elements:

Instruction - a specific task or instruction you want the model to perform

Context - external information or additional context that can steer the model to better responses

Input Data - the input or question that we are interested to find a response for

Output Indicator - the type or format of the output.
```
Reference: [Prompt Engineering Guide](https://www.promptingguide.ai/introduction/elements)

Recommended reading: 
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI's Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/prompt-engineering)

## What are some common prompt engineering techniques?

1. Zero-shot Prompting
2. Few-shot Prompting
3. Chain-of-Thought Prompting
4. Self-Consistency
5. Generate Knowledge Prompting
6. Prompt Chaining
7. Tree of Thoughts
8. Retrieval Augmented Generation
9. ReAct

Reference: [Prompt Engineering Guide](https://www.promptingguide.ai/techniques)

Recommended reading: 
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI's Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/prompt-engineering)

## Explain the concept of In-context Learning
In-context learning is a very intuitive and easy to understand learning paradigm in Natural Language Processing. It encompasses concepts such as few-shot learning. It can be as easy as providing a few examples of the task you want the model to perform, and the model will learn from those examples and generate responses accordingly.

Recommended Reading:
- [Towards Understanding In-context Learning from COS 597G (Fall 2022): Understanding Large Language Models at Princeton University](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec07.pdf)
- [Understanding In-Context Learning](https://ai.stanford.edu/blog/understanding-incontext/)

## When does In-context Learning fail?
It has been shown that In-context Learning can only emerge when the models are scaled to a certain size, and when the models are trained on a diverse set of tasks. In-context learning can fail when the model is not able to perform complex reasoning tasks.

Recommended Reading:
- [Few-shot Prompting by Prompt Engineering Guide](https://www.promptingguide.ai/techniques/fewshot)

## What would be a good methodology for designing prompts for a specific task?
This is a very broad question, but the following will help you form a basic understanding of how to design prompts for a specific task:
- [Best Practices for Prompt Engineering from OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [General Tips for Designing Prompts](https://www.promptingguide.ai/introduction/tips)

## Explain the concept of hallucination in the context of LLMs
```
The term describes when LLMs produce text that is incorrect, makes no sense, or is unrelated to reality
```
Reference: [LLM Hallucination—Types, Causes, and Solution by Nexla](https://nexla.com/ai-infrastructure/llm-hallucination/)

Recommended Reading:
- [Hallucination (Artificial Intelligence) - Wikipedia](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
- [Hallucination is Inevitable: An Innate Limitation of Large Language Models](https://arxiv.org/pdf/2401.11817)
- [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232)

## What prompt engineering concept is known to enhance reasoning capabilities in LLMs?
The concept of Chain-of-Thought Prompting is known to enhance reasoning capabilities in LLMs. This technique involves breaking down a complex task into a series of simpler tasks, and providing the model with the intermediate outputs of each task to guide it towards the final output.

Recommended Reading:
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Chain-of-Thought Prompting by Prompt Engineering Guide](https://www.promptingguide.ai/techniques/cot)

## What is a common design pattern for grounding LLM answers in facts?
Retrieval Augmented Generation (RAG) is a common design pattern for grounding LLM answers in facts. This technique involves retrieving relevant information from a knowledge base and using it to guide the generation of text by the LLM.

## Explain the intuition and methodology behind Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation (RAG) is composed of two main component:
- A retriever: This component is responsible for retrieving relevant information from a knowledge base given a query.
- A generator: This component is responsible for generating text based on the retrieved information.

The intuition behind RAG is that by combining the strengths of retrieval-based and generation-based models, we can create a system that is capable of generating text that is grounded in facts, thus limiting hallucination.

RAG is often the go-to technique for answering complex questions based on a knowledge base, as it allows the model to leverage external information to provide more accurate and informative answers. It is not always feasible to fine-tune a model on proprietary data, and RAG provides a way to incorporate external knowledge without the need for fine-tuning.

## Provide a high-level overview of the steps involved in implementing a full solution that utilizes RAG to answer a complex question based on a knowledge base
A full solution that utilizes RAG to answer a complex question based on a knowledge base would involve the following steps:
- Data Ingestion: documents or data streams that compromise the knowledge base are ingested into a data pipeline and processed in a way that is suitable for retrieval.
- Indexing: the processed data is indexed in a way that allows for efficient retrieval.
- Retrieval: given a query, the retriever component retrieves relevant information from the knowledge base.
- Generation: the generator component generates text based on the retrieved information.
- Post-processing: the generated text is post-processed to ensure factuality and integrity.

## Discuss the case of RAG vs. Full Fine-tuning
Recommended Reading (both sides of the argument):
- [Fine-Tuning vs. Retrieval Augmented Generation (RAG): Tailoring Large Language Models to Your Needs](https://www.linkedin.com/pulse/fine-tuning-vs-retrieval-augmented-generation-rag-tailoring-liz-liu/?trackingId=iQpGPevfTpG5eIb5v1%2BaJA%3D%3D)
- [Enhancing LLMs with Retrieval Augmented Generation](https://scale.com/blog/retrieval-augmented-generation-to-enhance-llms?utm_source=linkedin&utm_medium=organic-social&utm_campaign=rag-blog)
- [Post by Justin Zhao, Foudning Engineer @ Predibase](https://www.linkedin.com/posts/justin-zhao_we-keep-hearing-questions-about-fine-tuning-activity-7159251147076067328-flhR?utm_source=share&utm_medium=member_desktop)
- [Musings on building a Generative AI product - Linkedin Engineering](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)

