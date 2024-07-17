This a project I'm working on right now, I'm trying to compile a list of questions and answers for Generative AI interviews.

I'm using this [reference](https://www.masteringllm.com/course/llm-interview-questions-and-answers) as the base, credit to them for compiling it, however, I am taking _a lot_ of liberties with editing the questions, as well as the answers, they are completely my own.

_Note:_ I'm trying to keep the answers I write myself to a minimum, since I am in no way or form an authoritative source on this topic. I will be providing references to the best of my ability. I refrained from adding any sort of visual aid for readability and to keep the complexity of maintenance to a minimum. The resources and references I cite contain a wealth of information, mostly with visuals.

I plan to expand this to Generative AI in general, not just for language, covering everything from diffusion models up to vision-language models.
Once I get the basic structure down and I'm happy with the preliminary results, I will work on establishing an efficient methodology for contributing to this repository, and then I will open it up for contributions, but for now, I want to keep it simple and focused.

## Preamble
**Important:** 
> I think it might be necessary to clarify that the answers I provide, regardless if they are my own writeup or if I'm citing a source, are not in any way or form definitive, what I'm trying to do is get you started on the right path, and give you a general idea of what to expect, you should definitely read any and all resources I provide, and then some. If you want this to be your last stop, this is the wrong place for you. This is where it starts.

Also, if you're just getting started, my one and only piece of advice is:
> Get comfortable reading papers, because they _never_ end.

Paper on How to Read a Paper: [How to Read a Paper](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)

## Table of Contents

[1. LLM and Prompting Basics](#llm-and-prompting-basics)    

[2. Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)  

[3. Chunking Strategies](#chunking-strategies)  

[4. Embedding Models for Retrieval](#embedding-models-for-retrieval)    

[5. Vector Retrieval, Databases and Indexes](#vector-retrieval-databases-and-indexes)

[6. Advanced Search Algorithms](#advanced-search-algorithms)

[7. Language Model Internal Workings](#language-model-internal-workings)

## LLM and Prompting Basics

### What is the difference between Predictive/Discriminative and Generative AI Models?
Let's consider a dataset, where each data point represents a cat. Let's pass it through to each type of model, and see how they differ:
- **Predictive/Discriminative Model**: This type of model will either try to discriminate between different data points, by virtue of a decision boundary, and a mapping function, or by predicting the probability of a class label given the model's ability to learn patterns from the data.
- **Generative Model**: This type of model will try to generate new data points, by learning the underlying distribution of the data given all the data points in the dataset.

### What is a Large Language Model (LLM)?
Let's build the definition of a Large Language Model (LLM) from the ground up:
- **Language Models (LMs)**: These are probabilistic models that learn from natural language, they can range from simple bayesian models to more complex neural networks.
- **Large**: Consider all the data available, a literal dump of the internet, enough compute to process and learn from it, and a model that can handle the complexity of the data.
- **Large Language Model**: The amalgamation of the two, a model that can learn from a large amount of data, and generate text that is coherent and human-like.

Further reading:
[Common Crawl](https://commoncrawl.org/)

### How are Large Language Models trained?
Large Language Models are often trained in multiple stages, these stages are often named pretraining, fine-tuning, and alignment. 
#### Pretraining: 
The purpose of this stage is to expose the model to _all of language_, in an unsupervised manner, it is often the most expensive part of training, and requires a lot of compute. Pretraining is often done on something like the [Common Crawl](https://commoncrawl.org/) dataset, processed versions of the dataset such as [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [RedPajama](https://github.com/togethercomputer/RedPajama-Data) are often used for pretraining.
To facilitate this broad type of learning, there exists multiple training tasks we can use, such as Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and more.

Mask Language Modeling is based of the [Cloze Test](https://en.wikipedia.org/wiki/Cloze_test), where we mask out a word in a sentence, and ask the model to predict it. Similar to a fill in the blank test. It differs from asking the model to predict the next word in a sentence, as it requires the model to understand the context of the sentence, and not just the sequence of words.

Next Sentence Prediction is a task where the model is given two sentences, and it has to predict if the second sentence follows the first one. As simple as it sounds, it requires the model to understand the context of the first sentence, and the relationship between the two sentences.

An excellent resource to learn more about these tasks is the [BERT paper](https://arxiv.org/abs/1810.04805).

#### Fine-tuning:
This stage is much simpler than pretraining, as the model has already learned a lot about language, and now we just need to teach it about a specific task. All we need for this stage is the input data (prompts) and the labels (responses). 

#### Alignment:
This stage is often the most crucial and complex stage, it requires the use of separate reward models, the use of different learning paradigms such as Reinforcement Learning, and more. 

This stage mainly aims to align the model's predictions with the human's preferences. This stage often interweaves with the fine-tuning stage. Essential reading for this stage is the [InstructGPT paper](https://arxiv.org/abs/2203.02155), this paper introduced the concept of Reinforcement Learning from Human Feedback (RLHF) which uses [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

Other methods of Aligning the model's predictions with human preferences include:
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

### What is a token in the context of LLMs?
Tokens are the smallest unit of text that the model can understand, they can be words, subwords, or characters.

### What are tokenizers in the context of LLMs?
Tokenizers are responsible for converting text into tokens, they can be as simple as splitting the text by spaces, or as complex as using subword tokenization. The choice of tokenizer can have a significant impact on the model's performance, as it can affect the model's ability to understand the context of the text.

Some common tokenizers include:
- [Byte Pair Encoding (BPE)](https://aclanthology.org/P16-1162.pdf) 
- [WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [SentencePiece](https://arxiv.org/abs/1808.06226)

Recommended reading (and watching):
- [Summary of Tokenizers by HuggingFace](https://huggingface.co/docs/transformers/en/tokenizer_summary)
- [Let's build the GPT Tokenizer by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)

### How do you estimate the cost of running an API based / closed source LLM vs. an open-source LLMs (self-hosted)?
This is a very loaded question, but here are some resources to explore this topic further:
- [Using LLMs for Enterprise Use Cases: How Much Does It Really Cost?](https://www.titanml.co/resources/using-llms-for-enterprise-use-cases-how-much-does-it-really-cost)
- [The Challenges of Self-Hosting Large Language Models](https://www.titanml.co/resources/the-challenges-of-self-hosting-large-language-models)
- [The Case for Self-Hosting Large Language Models](https://www.titanml.co/resources/the-case-for-self-hosting-large-language-models)
- [Exploring the Differences: Self-hosted vs. API-based AI Solutions](https://www.titanml.co/resources/exploring-the-differences-self-hosted-vs-api-based-ai-solutions)

### What are the different parameters that can be tuned in LLMs during inference?
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

### What are the different decoding strategies for picking output tokens?
Decoding strategies are used to pick the next token in the sequence, they can range from simple greedy decoding to more complex sampling strategies. 

Some common decoding strategies include:
- Greedy Decoding
- Beam Search

Newer decoding strategies include Speculative Decoding (assisted decoding) which is a wild concept, it involves using a candidate tokens from a smaller (thus faster) model to generate a response from a bigger model very quickly.

Recommended reading:
- [Text generation strategies by HuggingFace](https://huggingface.co/docs/transformers/generation_strategies)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192.pdf)
- [A Hitchhiker’s Guide to Speculative Decoding by Team PyTorch at IBM](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

### What are the stopping criteria for decoding in the context of LLMs?
In the decoding process, LLMs autoregressively generate text one token at a time. There are several stopping criteria that can be used to determine when to stop generating text. Some common stopping criteria include:
- **Maximum Length:** Stop generating text when the generated text reaches a certain length.
- **End of Sentence Token:** Stop generating text when the model generates an end of sentence token.
- **Stop Sequences:** Stop generating text when the model generates a predefined stop sequence.

### What are some elements that make up a prompt?
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

### What are some common prompt engineering techniques?

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

### Explain the concept of In-context Learning
In-context learning is a very intuitive and easy to understand learning paradigm in Natural Language Processing. It encompasses concepts such as few-shot learning. It can be as easy as providing a few examples of the task you want the model to perform, and the model will learn from those examples and generate responses accordingly.

Recommended Reading:
- [Towards Understanding In-context Learning from COS 597G (Fall 2022): Understanding Large Language Models at Princeton University](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec07.pdf)
- [Understanding In-Context Learning](https://ai.stanford.edu/blog/understanding-incontext/)

### When does In-context Learning fail?
It has been shown that In-context Learning can only emerge when the models are scaled to a certain size, and when the models are trained on a diverse set of tasks. In-context learning can fail when the model is not able to perform complex reasoning tasks.

Recommended Reading:
- [Few-shot Prompting by Prompt Engineering Guide](https://www.promptingguide.ai/techniques/fewshot)

### What would be a good methodology for designing prompts for a specific task?
This is a very broad question, but the following will help you form a basic understanding of how to design prompts for a specific task:
- [Best Practices for Prompt Engineering from OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [General Tips for Designing Prompts](https://www.promptingguide.ai/introduction/tips)
- [Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4](https://arxiv.org/abs/2312.16171v2)
- [Prompting Fundamentals and How to Apply them Effectively by Eugene Yan (Amazon)](https://eugeneyan.com/writing/prompting/?es_id=3eb4c98a3b)

Alternatively, newer research directions investigate using an algorithmic way of optimizing the prompts, this has been explored extensively in the [DSPy](https://github.com/stanfordnlp/dspy) package, which provides the means to do this, their work is also published in this [paper](https://arxiv.org/abs/2212.14024).
### What LLM framework produces the best outputs?
There is no answer for this question, I put it as an excuse to link this reference:
- [Fuck You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/)

### Suppose you want an LLM to output JSON or YAML or any other structure, how would you ensure the output is parsable?
There are multiple methods to get LLMs to generate structured outputs that are parsable every time, common methods depends on the concept of function calling in LLMs.

Recommended Reading and Viewing:
- [Instructor](https://github.com/jxnl/instructor)
- [LlamaIndex Structured Output](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/)
- [OpenAI JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode)

### Explain the concept of hallucination in the context of LLMs
```
The term describes when LLMs produce text that is incorrect, makes no sense, or is unrelated to reality
```
Reference: [LLM Hallucination—Types, Causes, and Solution by Nexla](https://nexla.com/ai-infrastructure/llm-hallucination/)

Recommended Reading:
- [Hallucination (Artificial Intelligence) - Wikipedia](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
- [Hallucination is Inevitable: An Innate Limitation of Large Language Models](https://arxiv.org/abs/2401.11817)
- [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/abs/2311.05232)

### What prompt engineering concept is known to enhance reasoning capabilities in LLMs?
The concept of Chain-of-Thought Prompting is known to enhance reasoning capabilities in LLMs. This technique involves breaking down a complex task into a series of simpler tasks, and providing the model with the intermediate outputs of each task to guide it towards the final output.

Recommended Reading:
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Chain-of-Thought Prompting by Prompt Engineering Guide](https://www.promptingguide.ai/techniques/cot)

## Retrieval Augmented Generation (RAG)

### What is a common design pattern for grounding LLM answers in facts?
Retrieval Augmented Generation (RAG) is a common design pattern for grounding LLM answers in facts. This technique involves retrieving relevant information from a knowledge base and using it to guide the generation of text by the LLM.

Recommended Reading:
- [Patterns for Building LLM-based Systems & Products by Eugene Yan (Amazon)](https://eugeneyan.com/writing/llm-patterns/)
- [LLM From the Trenches: 10 Lessons Learned Operationalizing Models at GoDaddy](https://www.godaddy.com/resources/news/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy)
- [What We’ve Learned From A Year of Building with LLMs](https://applied-llms.org/)

### Explain the intuition and methodology behind Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation (RAG) is composed of two main component:
- **A retriever:** This component is responsible for retrieving relevant information from a knowledge base given a query.
- **A generator:** This component is responsible for generating text based on the retrieved information.

The intuition behind RAG is that by combining the strengths of retrieval-based and generation-based models, we can create a system that is capable of generating text that is grounded in facts, thus limiting hallucination.

RAG is often the go-to technique for answering complex questions based on a knowledge base, as it allows the model to leverage external information to provide more accurate and informative answers. It is not always feasible to fine-tune a model on proprietary data, and RAG provides a way to incorporate external knowledge without the need for fine-tuning.

### Provide a high-level overview of the steps involved in implementing a full solution that utilizes RAG to answer a complex question based on a knowledge base
A full solution that utilizes RAG to answer a complex question based on a knowledge base would involve the following steps:
- **Data Ingestion:** documents or data streams that compromise the knowledge base are ingested into a data pipeline and processed in a way that is suitable for retrieval.
- **Indexing:** the processed data is indexed in a way that allows for efficient retrieval.
- **Retrieval:** given a query, the retriever component retrieves relevant information from the knowledge base.
- **Generation:** the generator component generates text based on the retrieved information.
- **Post-processing:** the generated text is post-processed to ensure factuality and integrity.

### Discuss the case for RAG vs. Full Fine-tuning
This is a very loaded question, but here are some resources to explore this topic further:
- [Fine-Tuning vs. Retrieval Augmented Generation (RAG): Tailoring Large Language Models to Your Needs](https://www.linkedin.com/pulse/fine-tuning-vs-retrieval-augmented-generation-rag-tailoring-liz-liu/?trackingId=iQpGPevfTpG5eIb5v1%2BaJA%3D%3D)
- [Enhancing LLMs with Retrieval Augmented Generation](https://scale.com/blog/retrieval-augmented-generation-to-enhance-llms?utm_source=linkedin&utm_medium=organic-social&utm_campaign=rag-blog)
- [Post by Justin Zhao, Founding Engineer @ Predibase](https://www.linkedin.com/posts/justin-zhao_we-keep-hearing-questions-about-fine-tuning-activity-7159251147076067328-flhR?utm_source=share&utm_medium=member_desktop)
- [Musings on building a Generative AI product - Linkedin Engineering](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)
- [RAG vs. Fine-tuning by Armand Ruiz, VP of Product, AI Platform @ IBM](https://www.linkedin.com/posts/armand-ruiz_rag-vs-fine-tuning-its-not-an-eitheror-activity-7202987364254638081-YpvE/?utm_source=share&utm_medium=member_android)
- [Patterns for Building LLM-based Systems & Products by Eugene Yan (Amazon)](https://eugeneyan.com/writing/llm-patterns/)
- [What We’ve Learned From A Year of Building with LLMs](https://applied-llms.org/)
- [LLM From the Trenches: 10 Lessons Learned Operationalizing Models at GoDaddy](https://www.godaddy.com/resources/news/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy)

## Chunking Strategies

### Explain the concept of chunking and its importance to RAG systems
Chunking text is the process of breaking down a large piece of text into smaller, more manageable chunks. In the context of RAG systems, chunking is important because it allows the retriever component to efficiently retrieve relevant information from the knowledge base. By breaking down the query into smaller chunks, the retriever can focus on retrieving information that is relevant to each chunk, which can improve the accuracy and efficiency of the retrieval process.

During the training of embedding models, which are often used as retrievers, positive and negative pairs of text are used to indicate what pieces of text correspond to each other, examples include the titles, headers and subheaders on a Wikipedia page, and their corresponding paragraphs, reddit posts and their top voted comments, etc.

A user query is often embedded, and an index is queried, if the index had entire documents contained within it to be queried for top-k hits, a retriever would not be able to return the most relevant information, as the documents to be queried would be too large to comprehend.

To summarize, we chunk text because:
- We need to break down large pieces of text into smaller, more manageable chunks, where we ideally wish to have each chunk contain defined pieces of information we can query.
- Embedding models often have fixed context lengths, we cannot embed an entire book.
- Intuitively, when we search for information, we know the book we want to use as reference (corresponding to an index here), we'd use chapters and subchapters (our chunks) to find the information we need.
- Embedding models compress semantic information into a lower dimensional space, as the size of the text increases, the amount of information that is lost increases, and the model's ability to retrieve relevant information decreases.

### Explain the intuition behind chunk sizes using an example
Suppose we have a book, containing 24 chapters, a total of 240 pages.
This would mean that each chapter contains 10 pages, and each page contains 3 paragraphs.
Let's suppose that each paragraph contains 5 sentences, and each sentence contains 10 words.
In total, we have: 10 * 5 * 3 * 10 = 1500 words per chapter. 
We also have 1500 * 24 = 36000 words in the entire book.
For simplicity, our tokenizer is a white space tokenizer, and each word is a token.

We know that at most, we have an embedding model capable of embedding 8192 tokens:
- If we were to embed the entire book, we would have to chunk the book into 5 chunks. Each chunk would contain 5 chapters, with 7200 tokens. This is a tremendous amount of information to embed, and the model would not be able to retrieve relevant information efficiently.
- We can embed each chapter individually, this would mean that each chapter would yield 1500 tokens, which is well within the model's capacity to embed. But we know that chapters contain multiple topics, and we would not be able to retrieve the most relevant information.
- We can embed each page, resulting in 450 tokens per page, this is a good balance between the two extremes, as pages often contain a single topic, and we would be able to retrieve the most relevant information, however, what if the information we need is spread across multiple pages?
- We can embed each paragraph individually, this would mean that each paragraph would yield 150 tokens, which is well within the model's capacity to embed. TParagraphs often contain a single topic, and we would be able to retrieve the most relevant information, however, what if the flow of information is not linear, and the information we need is spread across multiple paragraphs, and we need to aggregate it?
- We can embed each sentence individually, but here we risk losing the context of the paragraph, and the model would not be able to retrieve the most relevant information.

All of this is to illustrate that there is no fixed way to chunk text, and the best way to chunk text is to experiment and see what works best for your use case.

### What are the different chunking strategies used in RAG systems and how do you evaluate your chunking strategy?
An authoritative source on this topic is the excellent [notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) and accompanying [video](https://www.youtube.com/watch?v=8OJC21T2SL4) by [Greg Kamradt](https://www.youtube.com/@DataIndependent), in which they explain the different levels of text splitting. 

The notebook also goes over ways to evaluate and visualize the different levels of text splitting, and how to use them in a retrieval system.

Recommended Viewing:
- [ChunkViz: A Visual Exploration of Text Chunking](https://chunkviz.up.railway.app/)
- [RAGAS: An Evaluation Framework for Retrieval Augmented Generation](https://github.com/explodinggradients/ragas)

## Embedding Models for Retrieval

### Explain the concept of a vector embedding
Vector embeddings are the mapping of textual semantics into an N-dimensional space where vectors represent text, within the vector space, similar text is represented by similar vectors.

Recommended Reading:
- [What are Vector Embeddings?](https://www.elastic.co/what-is/vector-embedding)

### What are embedding models?
Embedding models are Language Models trained for the purpose of vectorizing text, they are often BERT derivatives, and are trained on a large corpus of text to learn the semantics of the text, recent trends however also show it is possible to use much larger language models for this purpose such as Mistral or Llama.

Recommended Reading and Viewing:
- [Quickstart for SentenceTransformers](https://www.sbert.net/docs/quickstart.html)
- [llm2vec](https://github.com/McGill-NLP/llm2vec)
- [NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/abs/2405.17428)

### How do embedding models work in the context of systems with LLMs?
Embedding models are often used as retrievers, to utilize their retrieval capabilities, semantic textual similarity is used where in vectors produced by the models are measured in similarity using metrics such as dot product, cosine similarity, etc.

- [Retriever Documentation by LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)

### What are the different types of text embeddings?
- Dense Multi-vectors (e.g. ColBERT)
- Dense Single-vectors (e.g. BERT with Pooling)
- Sparse Single-vectors (e.g. BM25 or SPLADE)

Recommended Reading:
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)

### How do you train an embedding model?
Embeddings models are trained with contrastive loss, ranging from simple contrastive loss and up to more complex loss functions such as InfoNCE and Multiple Negative Ranking Loss. A process known as hard negative mining is also utilized during training as well.

Recommended Reading:
- [Contrastive Representation Learning by Lilian Weng (OpenAI)](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SentenceTransformer Losses Documentation](https://www.sbert.net/docs/sentence_transformer/loss_overview.html)
- [Hard Negative Mining Used by BGE Text Embedding Models](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/hn_mine.py)

### Explain the concept of contrastive learning in the context of embedding models
Contrastive learning is a technique used to train embedding models, it involves learning to differentiate between positive and negative pairs of text. The model is trained to maximize the similarity between positive pairs and minimize the similarity between negative pairs.

Recommended Reading:
- [SentenceTransformers Losses](https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses)
- [Contrastive Representation Learning by Lilian Weng (OpenAI)](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SentenceTransformer Losses Documentation](https://www.sbert.net/docs/sentence_transformer/loss_overview.html)
- [Hard Negative Mining Used by BGE Text Embedding Models](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/hn_mine.py)


### Explain the intuition behind single vector dense representations
Single vector dense representations are often the norm in text embedding models, they're usually produced by pooling the contextualized embeddings after a forward pass from the model, pooling techniques include mean pooling, max pooling, and CLS token pooling. The intuition behind single vector dense representations is that they are simple to implement and can be used for a wide range of tasks, as well as ease of indexing and retrieval. Dense representations are also able to capture the semantics of the text, and are often used in second stage ranking.

### Explain the intuition behind multi vector dense representations
Multi vector dense representations have shown to produce superior results to single vector dense representations, they are produced by skipping the pooling step and using the contextualized embeddings in the form of a matrix, the query and document embeddings are then used to calculate the similarity between the two, models such as ColBERT have shown to produce superior results to single vector dense representations. 
An operator such as MaxSim is used to calculate the similarity between the query and document embeddings. The intuition behind multi vector dense representations is that they are able to capture more information about the text, and produce better results than single vector dense representations, models such as ColBERT also offer the ability to precompute document embeddings, allowing for very efficient retrieval. Dense representations are also able to capture the semantics of the text, and are often used in second stage ranking.

Recommended Reading:
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

### Explain the intuition behind sparse representations
Sparse text representations are the oldest form of vector space models in information retrieval, they are usually based on TF-IDF derivatives and algorithms such as BM25, and remain a baseline for text retrieval systems. Their sparsity stems from the fact that the dimension of the embeddings often corresponds to the size of the vocabulary. The intuition behind sparse representations is that they are explainable, computationally efficient, easy to implement and extremely efficient for indexing and retrieval. Sparse representation also focus on lexical similarity, and are often used in first stage ranking.

Recommended Reading:
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)

### Elaborate on why sparse text embeddings are more efficient than dense text embeddings
Sparse text embeddings allow for the use of inverted indices during retrieval.

Recommended Reading:
- [Inverted Indexes](https://en.wikipedia.org/wiki/Inverted_index)

### How do you benchmark the performance of an embedding model?
Metrics for benchmarking the performance of an embedding model include:
- Recall
- Precision
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

Recommended Reading and Viewing:
- [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- [Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [ir-measures](https://ir-measur.es/en/latest/measures.html)

### How do you pick an embedding model for a specific use case?
Picking an embedding model could be a pivotal factor in the performance of your retrieval system, and careful consideration should be taken when choosing one. It is a broad process that involves experimentation, and the following resources will help you make an informed decision:
- [Step-by-Step Guide to Choosing the Best Embedding Model for Your Application](https://weaviate.io/blog/how-to-choose-an-embedding-model)

Recommended Viewing:
- [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- [Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)

## Vector Retrieval, Databases and Indexes

### Quintessential Recommended Reading on Vector Retrieval
- [Foundations of Vector Retrieval by Sebastian Bruch](https://arxiv.org/abs/2401.09350)
- [Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/)

### What is a vector database?
A vector database is a database that is optimized for storing and querying vector data. It allows for efficient storage and retrieval of vector embeddings, and is often used in applications that require semantic similarity search.
Vector databases are a new paradigm that has emerged as part of the tech stack needed to keep up with the demands of GenAI applications.

Recommended Viewing:
- [Qdrant](https://qdrant.tech/)
- [Milvus](https://milvus.io/)
- [Chroma](https://www.trychroma.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Weaviate](https://weaviate.io/)
- [Vespa](https://vespa.ai/)

### How are vector databases different from traditional databases?
Traditional databases are optimized for storing and querying structured data, such as text, numbers, and dates. They are not designed to handle vector data efficiently. Vector databases, on the other hand, are specifically designed to store and query vector data. They use specialized indexing techniques and algorithms to enable fast and accurate similarity search such as quantization and clustering of vectors.

### How does a vector database work?
A vector database usually contains indexes of vectors, these indexes contain matrices of vector embeddings, ordered in such a way that they can be queried efficiently. When a query is made, either text or a vector embedding is provided as input, in the case of text, it is embedded, and the vector database will query the appropriate index to retrieve the most similar vectors based on distance metrics. Usually, the vectors are compared using metrics such as cosine similarity, dot product, or Euclidean distance. Vectors also relate to a dictionary of metadata that could contain information such as the document ID, the document title, the corresponding text and more.

### What are the different strategies for searching in a vector database or index?
Search strategies in vector databases include:
- **Exhaustive Search:** This strategy involves comparing the query vector with every vector in the database to find the most similar vectors.
- **Approximate Search:** This strategy involves using approximate algorithms such as Hierarchal Navigable Small Worlds (HNSW) to find the most similar vectors. At the time of indexing, a graph is built, and the query vector is traversed through the graph to find the most similar vectors.

Recommended Reading:
- [Foundations of Vector Retrieval by Sebastian Bruch, Part II Retrieval Algorithms](https://arxiv.org/abs/2401.09350)
- [Approximate Near Neighbor Search](https://www.cs.toronto.edu/~anikolov/CSC473W19/Lectures/LSH.pdf)
- [Faiss Index Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Hierarchical Navigable Small Worlds](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)

### Explain the concept of clustering in the context of vector retrieval, and its relation to the search space
Once the vectors are indexed, they are often clustered to reduce the search space, this is done to reduce the number of vectors that need to be compared during the search process. Clustering is done by grouping similar vectors together, and then indexing the clusters. When a query is made, the search is first performed at the cluster level, and then at the vector level within the cluster. Algorithms such as K-means are often used for clustering.

Recommended Reading:
- [UNum USearch Documentation](https://github.com/unum-cloud/usearch#clustering)
- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Faiss Index Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)
- [Foundations of Vector Retrieval by Sebastian Bruch, Part II Retrieval Algorithms, Chapter 7 Clustering](https://arxiv.org/abs/2401.09350)
- [Hierarchical Navigable Small Worlds](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)

### Compare between various vector databases and indexes based on a set of criteria, and decide which one is best for a given use case
This is obviously a very loaded question, but here are some resources to explore this topic further:
- [Vector Database Feature Matrix by Superlinked](https://superlinked.com/vector-db-comparison?utm_source=reddit&utm_medium=social&utm_campaign=vdb_launch)

### Explain the concept of quantization in the context of vectors
    Vector quantization, also called "block quantization" or "pattern matching quantization" is often used in lossy data compression. It works by encoding values from a multidimensional vector space into a finite set of values from a discrete subspace of lower dimension. 

Reference: [Vector Quantization](https://en.wikipedia.org/wiki/Vector_quantization)

### Explain the concept of locality-sensitive hashing (LSH) within the context of vector retrieval
    One general approach to LSH is to “hash” items several times, in such a way that similar items are more likely to be hashed to the same bucket than dissimilar items are.

Reference: [Mining of Massive Datasets, 3rd Edition, Chapter 3, Section 3.4.1](http://mmds.org/)

Recommended Reading:
- [Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [Locality Sensitive Hashing (LSH): The Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)

### Explain the concept of product quantization within the context of vector retrieval
    In short, PQ is the process of:
    - Taking a big, high-dimensional vector,
    - Splitting it into equally sized chunks — our subvectors,
    - Assigning each of these subvectors to its nearest centroid (also called reproduction/reconstruction values),
    - Replacing these centroid values with unique IDs — each ID represents a centroid

Reference: [Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/)

Recommended Reading:
- [Product Quantization for Nearest Neighbor Search](https://www.researchgate.net/publication/47815472_Product_Quantization_for_Nearest_Neighbor_Search)

### Explain the concept of inverted indices within the context of vector retrieval
    The Inverted File Index (IVF) index consists of search scope reduction through clustering.

Reference: [Inverted File Index](https://www.pinecone.io/learn/series/faiss/vector-indexes/#Inverted-File-Index)

Recommended Reading:
- [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes)

### Explain the concept of Hierarchical Navigable Small Worlds (HNSW) within the context of vector retrieval
Hierarchical Navigable Small Worlds (HNSW) is often considered the state-of-the-art in vector retrieval, it is a graph-based algorithm that builds a graph of the vectors, and uses it to perform approximate nearest neighbor search.

Recommended Reading:
- [Hierarchical Navigable Small Worlds - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- [Hierarchical Navigable Small Worlds - Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/hnsw/)

### Explain some of the distance and similarity metrics used in vector retrieval
Distance and similarity metrics used in vector retrieval include:
- [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
- [Dot Product](https://en.wikipedia.org/wiki/Dot_product)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
- [Dice-Sørensen Coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient)
- [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
- [Haversine Distance](https://en.wikipedia.org/wiki/Haversine_formula)

Recommended Viewing:
- [Fast Vector Similarity](https://github.com/Dicklesworthstone/fast_vector_similarity)

## Advanced Search Algorithms

### Explain some of the architectures and patterns in retrieval systems and semantic search in the context of LLM systems
This is a very active research topic, and no authoritative source exists, but here are some resources to explore this topic further:
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856)
- [Levels of Complexity: RAG Applications](https://github.com/jxnl/n-levels-of-rag)
- [Beyond the Basics of Retrieval for Augmenting Generation (w/ Ben Clavié)](https://www.youtube.com/watch?v=0nA5QG3087g)

It is also worth noting that search, retrieval and reranking systems are built on established patterns and architectures in the fields of information retrieval, recommendation systems, and search engines.

Some system architectures you might want to explore include:
- [Twitter Algorithm](https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm)
- [Code for Twitter Algorithm](https://github.com/twitter/the-algorithm)
- [Representation Learning for Recommender Systems](https://oars-workshop.github.io/2021/andrew.pdf)

### How do you achieve good search in large scale systems?
Achieving good search in large-scale systems involves a combination of efficient indexing, retrieval, and ranking techniques. Some strategies to achieve good search in large-scale systems include:
- Using a lightweight retriever to generate a set of candidate documents algorithms such as [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and keyword search are often used for this purpose.
- Using a more complex retriever to generate a set of candidate documents, models such as [ColBERT](https://arxiv.org/abs/2004.12832) and [BGE-M3](https://arxiv.org/abs/2402.03216) are often used for this purpose.
- Using a reranker to re-rank the candidate documents generated by the retriever, models such as [mixedbread-ai/mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) and [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) are often used for this purpose.
- Combining the results of multiple retrieval methodologies such as dense and sparse using [Reciprocal Rank Fusion](https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search)
- Implement query understanding and decomposition to break down complex queries into simpler sub-queries, and then recombine the results.
- Use metadata to filter results, no system works with just semantic and vector similarity, metadata is often used to filter results.

You might notice that the entire process is done in phases of increasing complexity, this is known as phased ranking or multistage retrieval.

Recommended Reading:
- [Phased Ranking by Vespa](https://docs.vespa.ai/en/phased-ranking.html)
- [Semantic Models for the First-stage Retrieval: A Comprehensive Review](https://arxiv.org/abs/2103.04831)
- [Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community](https://arxiv.org/abs/2307.09751)

But the most important aspect of achieving good search in large-scale systems is to experiment and iterate on your retrieval and ranking strategies, and to continuously monitor and evaluate the performance of your system.

Recommended Reading:
- [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- [Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [ir-measures](https://ir-measur.es/en/latest/measures.html)
- [Good Authority Figure on IR, one Tweet as a starting point](https://twitter.com/jobergum/status/1798640402427515293)

### Explain the concept behind keyword based search in the context of search systems
    BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document.

Reference: [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

### How are reranking models trained?
Reranking models are sequence classification models trained to take a pair of query and documents, and output raw similiarity scores. 

Recommended Reading, Viewing and Watching:
- [Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline](https://arxiv.org/abs/2101.08751)
- [SentenceTransformers Documentation on Cross-Encoders](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
- [Reranker Libarary](https://github.com/luyug/Reranker)
- [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/)
- [Training a Language Model for Reranking (RankZephyr)](https://www.youtube.com/watch?v=vtH93FB6nac)

### Explain the process for evaluating an end-to-end RAG system
Evaluating RAG systems requires experimenting with and evaluating the individual components of the system, such as the retriever, generator, and reranker.

Recommended Reading:
- [RAGAS: An Evaluation Framework for Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [Evaluating Retrieval Augmented Generation using RAGAS](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas)
- [Evaluating Retrieval Augmented Generation - a framework for assessment](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework)
- [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- [Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [ir-measures](https://ir-measur.es/en/latest/measures.html)
- [Good Authority Figure on IR, one Tweet as a starting point](https://twitter.com/jobergum/status/1798640402427515293)

## Language Model Internal Workings

_Note:_ from here on, I'll refrain from answering as much as I can, and just link papers and references, this part is arguably one of the more complex parts, so it requires a lot of reading and understanding.

### Explain the concept of attention (and self-attention) in LLMs
To understand attention, you'll need to be familiar with the Transformer architecture, and their predecessor architectures. Here are some resources to get you started:
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [An Introductory Survey on Attention Mechanisms in NLP Problems](https://arxiv.org/abs/1811.05544)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
- [Transformer Taxonomy](https://kipp.ly/transformer-taxonomy/)

### What are the disadvantages of the self-attention mechanism in LLMs?
The main bottleneck of self-attention is its quadratic complexity with respect to the sequence length.
To understand the disadvantages of self-attention, you'll need familiarize yourself with attention alternatives, the following will help you get started:
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Hyena Hierarchy: Towards Larger Convolutional Language Models (Paper)](https://arxiv.org/abs/2302.10866)
- [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)
- [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052)

### What is the difference between encoders, decoders, and encoder-decoder architectures in LLMs?
- [Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)

### Explain the concept of positional encoding in LLMs
There are multiple ways to encode positional information in LLMs, the most common way is to use sinusoidal positional encodings, known as absolute positional encodings. Other methods include relative positional encodings, and newer methods such as Rotary Positional Embeddings. Here are some resources to get you started:
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [Understanding positional encoding in Transformers](https://www.blopig.com/blog/2023/10/understanding-positional-encoding-in-transformers/)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### Explain the concept of KV Caching in LLM Inference
To understand KV Cache, you'll need to be familiar with the Transformer architecture and its limitations.

Recommended Reading:
- [Transformers Optimization: Part 1 - KV Cache](https://r4j4n.github.io/blogs/posts/kv/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis](https://arxiv.org/abs/2405.08944)
- [MiniCache: KV Cache Compression in Depth Dimension for Large Language Models](https://arxiv.org/abs/2405.14366)
- [GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527)

### Explain the concept of a mixture of experts (MoE) in LLMs
Mixture of experts is a type of architecture in LLMs, to understand how it works, you should go through the following resources, which cover the most prominent MoE models:
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- [Introducing DBRX: A New State-of-the-Art Open LLM](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
- [MegaBlocks](https://github.com/databricks/megablocks)
- [Grok-1](https://github.com/xai-org/grok-1)
- [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055)