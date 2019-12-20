# Transformers in NLP

In this note book we will be look in very close details of the famous **transformer** as proposed in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). To accompany the digestion, we will also be looking at how **attention mechanism** works in deep learning and finally implement transformer with **transfer learning** with FastAI library.

Acknowledgement, the illustration of mostly guided by the amazing blog post by **Jay Alammar**. Please go and check out his blog series in [this link](http://jalammar.github.io/illustrated-transformer/).

## Contents

[***Self-Attention***](https://github.com/Sylar257/Transformers-in-NLP#self-attention): how the self-attention block works in deep learning architectures

[***Positional Encoding***](https://github.com/Sylar257/Transformers-in-NLP#positional_encoding): representing the order of the sequence using positional encoding

[***Transformer***](https://github.com/Sylar257/GCP-production-ML-systems#adaptable_ml_system): high-level structure of the transformer

[***Implementation***](https://github.com/Sylar257/GCP-production-ML-systems#high_performance_ML_system): Implement transformer with transfer learning on IMDB sentiment analysis dataset



## Self-Attention

This is probably one of the most important concept to understand when come to learning Transformer.

Say that we want to translate this sentence to Chinese:

*"The cat can’t jump onto the table because it’s too tall”* 

What does 'it’ refers to in this sentence? Does it refer to the *cat* or the *table*? To human, this is such a simple question but not as simple to an algorithm.

At the high-level, **Self-attention** allows the algorithm to associate '*it*' to '*cat*'. As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

### Self-attention: step one

The first step of calculating self-attention is to create three vectors from each of the encoder’s input vectors (this would be the **embeddings** of the words). Hence, we create a **Query vector**, a **Key vector**, and a **Value vector** for each word. These vectors are created by *multiplying* the embeddings by three matrices that we trained during the *training process*.

![query-key-value](images\query-key-value.png)

What are the “query”, “key”, and “value” vectors?

They’re abstractions that are useful for calculating and thinking about **attention**. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

### Self-attention: step two

With “query”, “key”, and “value”, we can compute a **score** for each word in the *input sentence*. The score determines how much focus to place on other parts of the **input sentence** as we encode a word at the given position.

![attention-score](images\attention-score.png)

This "**Z**" score can be thought as to give us a score distribution that represents the attention the network has over all other word in the input sentence.

## Multi-headed attention

This is another mechanism introduced by the authors to further improve the performance of the attention layers. It has mainly two effects:

1. It expands the model’s ability to focus on different positions. Yes, in the example above, "**Z**" contains a little bit of every other encoding, but it could be *dominated* by the actual word itself (thanks to the *softmax*). It would be useful if we’re translating a sentence like *"The cat can’t jump onto the table because it’s too tall”* , when the algorithm knows what the word "it’s” is referring to.
2. It give the attention layer multiple "representation subspaces”. As we will see for the illustration below, with multi-headed attention we have not only one, but multiple sets of **Query/Key/Value** weight matrices (the **Transformer** uses **eight** attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

![attention-score](images\multiheaded-attention-QKV.png)

If we do the same **self-attention** calculation we outlined above, just eight different times with different weight matrices we end up with eight different **"Z”** matrices:

![multiheaded-attention-Z-score](images\multiheaded-attention-Z-score.png)

This leaves us with a bit of a challenge. The following layers is not expecting eight matrices — they are expecting a single matrix that contains a score for each word. So we need a way to *condense* these eight down to a single matrix. Solution: concatenate the matrices then use a another network to project it to the right shape:

