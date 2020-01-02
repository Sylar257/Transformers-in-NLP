# Transformers in NLP

In this note book we will be look in very close details of the famous **transformer** as proposed in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). To accompany the digestion, we will also be looking at how **attention mechanism** works in deep learning and finally implement transformer with **transfer learning** with FastAI library.

## Contents

[***Overview***](https://github.com/Sylar257/Transformers-in-NLP#overview): Current research trends, incentives of this repo, and our objectives

[***Self-Attention***](https://github.com/Sylar257/Transformers-in-NLP#self-attention): how the self-attention block works in deep learning architectures

[***Positional Encoding***](https://github.com/Sylar257/Transformers-in-NLP#positional_encoding): representing the order of the sequence using positional encoding

[***Residual connection***](https://github.com/Sylar257/Transformers-in-NLP#residuals): residuals are implemented for better learning efficiency and loss convergence

[***Transformer***](https://github.com/Sylar257/Transformers-in-NLP#Overall_structure): high-level structure of the transformer

[***Implementation***](https://github.com/Sylar257/Transformers-in-NLP#Implementation): Implement transformer with transfer learning on IMDB sentiment analysis dataset

## Overview

[*Jeremy Howard*](https://medium.com/@jeremyphoward) and [Sebastian Ruder](https://medium.com/@sebastianruder) first introduced **transfer learning** with NLP in the [ULMFiT paper](https://arxiv.org/pdf/1801.06146.pdf) and implemented the technique with the [FastAI](https://www.fast.ai/) library. 

For details of implementation of FastAI-NLP please refer to the [fast.ai course](https://course.fast.ai/videos/?lesson=4) or my two previous repos: [ULMFiT-Sentiment-Analysis](https://github.com/Sylar257/ULMFiT-Sentiment-Analysis) and [Non-English-language-NLP](https://github.com/Sylar257/Non-English-language-NLP).

In traditional NLP research, **transfer learning** was long reckoned as not applicable even after it’s great success in the computer vision research. For a long time, NLP was though as special and needs to be treated with special care even when it comes to deep learning solution implementation. Hence, nobody published anything about transfer learning for NLP until Jeremy et al. proved to the crowd that *"transfer learning is nothing specific to the computer vision domain”* and this whole kicked off. The recent NLP academic trend has began to shift towards transfer learning as the time-saving and learning efficiency boost is quite obvious.

Big name companies has continued to push this technique further with **transfer learning** applied with more sophisticated architectures at larger scales: Google(BERT, Transformer-XL, XLNet), Facebook(RoBERTa, XLM) and OpenAI(GPT, GPT-2). 

FastAI library itself comes only with [AWS-LSTM](https://arxiv.org/abs/1708.02182), [Transformer](https://arxiv.org/abs/1706.03762) and [Transformer-XL](https://arxiv.org/abs/1901.02860). If we want to implement more advanced models(such as BERT or RoBERTa) while still enjoy the fast-prototyping and massive optimization offered by `FastAI`, we need to learn to integrate the two.

More specifically, we are going integrate `FastAI` and the `transformers` library developed by *[Hugging Face](https://huggingface.co/)*, formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`. Now this library contains over 40 *state-of-the-art* **pre-trained** NLP models which we simply can’t pass on. It also come with essential utilities such as *tokenizer*, *optimizer* and *learning rate scheduler*. However, in this repo, we will use most of utility functions from `fastai` library as it’s a bit more optimized.



## Self-Attention

This is probably one of the most important concept to understand when come to learning Transformer.

Say that we want to translate this sentence to Chinese:

*"The cat can’t jump onto the table because it’s too tall”* 

What does 'it’ refers to in this sentence? Does it refer to the *cat* or the *table*? To human, this is such a simple question but not as simple to an algorithm.

At the high-level, **Self-attention** allows the algorithm to associate '*it*' to '*cat*'. As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

### Self-attention: step one

The first step of calculating self-attention is to create three vectors from each of the encoder’s input vectors (this would be the **embeddings** of the words). Hence, we create a **Query vector**, a **Key vector**, and a **Value vector** for each word. These vectors are created by *multiplying* the embeddings by three matrices that we trained during the *training process*.

![query-key-value](images/query-key-value.png)

What are the “query”, “key”, and “value” vectors?

They’re abstractions that are useful for calculating and thinking about **attention**. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

### Self-attention: step two

With “query”, “key”, and “value”, we can compute a **score** for each word in the *input sentence*. The score determines how much focus to place on other parts of the **input sentence** as we encode a word at the given position.

![attention-score](images/attention-score.png)

This "**Z**" score can be thought as to give us a score distribution that represents the attention the network has over all other word in the input sentence.

## Multi-headed attention

This is another mechanism introduced by the authors to further improve the performance of the attention layers. It has mainly two effects:

1. It expands the model’s ability to focus on different positions. Yes, in the example above, "**Z**" contains a little bit of every other encoding, but it could be *dominated* by the actual word itself (thanks to the *softmax*). It would be useful if we’re translating a sentence like *"The cat can’t jump onto the table because it’s too tall”* , when the algorithm knows what the word "it’s” is referring to.
2. It give the attention layer multiple "representation subspaces”. As we will see for the illustration below, with multi-headed attention we have not only one, but multiple sets of **Query/Key/Value** weight matrices (the **Transformer** uses **eight** attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

![attention-score](images/multiheaded-attention-QKV.png)

If we do the same **self-attention** calculation we outlined above, just eight different times with different weight matrices we end up with eight different **"Z”** matrices:

![multiheaded-attention-Z-score](images/multiheaded-attention-Z-score.png)

This leaves us with a bit of a challenge. The following layers is not expecting eight matrices — they are expecting a single matrix that contains a score for each word. So we need a way to *condense* these eight down to a single matrix. Solution: concatenate the matrices then use a another network to project it to the right shape:

![multiheaded-attention-score-condensation](images/multiheaded-attention-score-condensation.png)

## The overall picture of self-attention

Finally, when we put everything together.

![multiheaded-attention-overview](images/multiheaded-attention-overview.png)

# Positional_Encoding

One thing that’s missing from the model is the ability to locate the word’s position in the sentence.

To address this, the transformer adds a **vector** to each input embedding. These vectors follow a specific pattern that the model **learns**, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into **Q/K/V** vectors and during dot-product attention.

![positional-encoding](images/positional-encoding.png)

If we assumed the embedding has a dimensionality of *4*, the actual positional encodings would look like this:

![positional-encoding-example](images/positional-encoding-example.png)

# Residuals

Similar to the **ResNet**, in the transformer’s encoder architecture there are *residual connections*. In each encoder, there is a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step. If we visualize the architecture is looks like this:

![residual-connection](images/residual-connection.png)

**"X”** vector is the [positional encoding](https://github.com/Sylar257/Transformers-in-NLP#positional_encoding).

## Overall_structure

Of course, the layers are stacked for the **transformer**. We have **6-layer-stacking** for both *encoder* and *decoder* by the design of the [original paper](https://arxiv.org/abs/1706.03762). For illustration, if have a transformer of 2-layer-stacking:

![layer-stacking](images/layer-stacking.png)

## Implementation

You will find two Jupiter notebooks in this repo. 



When using the pre-trained models from `transformer` library, each model architecture need the following information:

1. A **model class** too lead/store a particular pre-trained model.
2. A **tokenizer class** to pre-process the data and make it compatible with our model of selection
3. A **configuration class** to load/store the configuration of a particular model

