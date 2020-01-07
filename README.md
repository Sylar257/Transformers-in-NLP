# Transformers in NLP

In this note book we will be look in very close details of the famous **transformer** as proposed in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). To accompany the digestion, we will also be looking at how **attention mechanism** works in deep learning and finally implement transformer with **transfer learning** with FastAI library.

## Contents

[***Overview***](https://github.com/Sylar257/Transformers-in-NLP#overview): Current research trends, incentives of this repo, and our objectives

[***Model selection***](https://github.com/Sylar257/Transformers-in-NLP#model_selection): Given so many high performance models, how do we choose which one to use?

[***Self-Attention***](https://github.com/Sylar257/Transformers-in-NLP#self-attention): How the self-attention block works in deep learning architectures

[***Positional Encoding***](https://github.com/Sylar257/Transformers-in-NLP#positional_encoding): Representing the order of the sequence using positional encoding

[***Residual connection***](https://github.com/Sylar257/Transformers-in-NLP#residuals): Residuals are implemented for better learning efficiency and loss convergence

[***Transformer***](https://github.com/Sylar257/Transformers-in-NLP#Overall_structure): High-level structure of the transformer

[***Implementation***](https://github.com/Sylar257/Transformers-in-NLP#Implementation): Implement transformer with transfer learning on IMDB sentiment analysis dataset

## Overview

[*Jeremy Howard*](https://medium.com/@jeremyphoward) and [Sebastian Ruder](https://medium.com/@sebastianruder) first introduced **transfer learning** with NLP in the [ULMFiT paper](https://arxiv.org/pdf/1801.06146.pdf) and implemented the technique with the [FastAI](https://www.fast.ai/) library. 

For details of implementation of FastAI-NLP please refer to the [fast.ai course](https://course.fast.ai/videos/?lesson=4) or my two previous repos: [ULMFiT-Sentiment-Analysis](https://github.com/Sylar257/ULMFiT-Sentiment-Analysis) and [Non-English-language-NLP](https://github.com/Sylar257/Non-English-language-NLP).

In traditional NLP research, **transfer learning** was long reckoned as not applicable even after it’s great success in the computer vision research. For a long time, NLP was though as special and needs to be treated with special care even when it comes to deep learning solution implementation. Hence, nobody published anything about transfer learning for NLP until Jeremy et al. proved to the crowd that *"transfer learning is nothing specific to the computer vision domain”* and this whole kicked off. The recent NLP academic trend has began to shift towards transfer learning as the time-saving and learning efficiency boost is quite obvious.

Big name companies has continued to push this technique further with **transfer learning** applied with more sophisticated architectures at larger scales: Google(BERT, Transformer-XL, XLNet), Facebook(RoBERTa, XLM) and OpenAI(GPT, GPT-2). 

FastAI library itself comes only with [AWS-LSTM](https://arxiv.org/abs/1708.02182), [Transformer](https://arxiv.org/abs/1706.03762) and [Transformer-XL](https://arxiv.org/abs/1901.02860). If we want to implement more advanced models(such as BERT or RoBERTa) while still enjoy the fast-prototyping and massive optimization offered by `FastAI`, we need to learn to integrate the two.

More specifically, we are going integrate `FastAI` and the `transformers` library developed by *[Hugging Face](https://huggingface.co/)*, formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`. Now this library contains over 40 *state-of-the-art* **pre-trained** NLP models which we simply can’t pass on. It also come with essential utilities such as *tokenizer*, *optimizer* and *learning rate scheduler*. However, in this repo, we will use most of utility functions from `fastai` library as it’s a bit more optimized.

## Code Implementation

You will find two Jupyter notebooks in this repo. In the [Transformer with no LM fine-tuning.ipynb](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Transformer%20with%20no%20LM%20fine-tuning.ipynb) we will implement transformer with `FastAI` library without fine-tuning the language model separately. This notebook follows strictly the guide provided by [Maximilien Roberti](https://towardsdatascience.com/@maximilienroberti). Details of his guide can be found both in his [Medium post](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2) as well as [Kaggle chanllange](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta). (Thumbs up for Maximilien Roberti)

The reason why there exists a second notebook is because in Maximilien’s [implementation](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta) he mainly replaced the AWD_LSTM model from `FastAI` by **RoBERTa** from `Hugging Face`. According to Jeremy’s [ULMFiT paper](https://arxiv.org/pdf/1801.06146.pdf), if we’d fine tuning the encoder of the **RoBERTa** model as a language model before constructing the classifier for sentiment analysis task the accuracy could be even better. Hence, in the [Implement various Transformers with FastAI.ipynb]([https://github.com/Sylar257/Transformers-in-NLP/blob/master/Implement%20various%20Transformers%20with%20FastAI.ipynb](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Implement various Transformers with FastAI.ipynb)) you will find the complete code for building two separate `databunch` for both language model and classification task as well as how can we apply *transfer learning* with `transformers` while taking advantage of the convenience of `FastAI` toolkit.

To benchmark our result we will be using the classic [IMDb sentiment analysis dataset](https://s3.amazonaws.com/fast-ai-nlp/imdb). Following the standard ULMFiT approach, in the last [repo](https://github.com/Sylar257/ULMFiT-Sentiment-Analysis), we were able to reach **94.7%** accuracy which is slightly better than the state-of-the-art result in 2017 (94.1% accuracy). Now let’s challenge ourselves to push these results even further by implementing more advanced skills.

In this repository you will find everything you need to incorporate transformers as the base architecture when using the `FastAI` framework. I will also try to provide you with the essential explanations of why we would make those customizations so that when you choose a different architecture you can replicate the process.

## Model_selection

BERT, GPT, GPT-2, Transformer-XL, XLNet, XLM, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, the list goes on. There are so many good performing NLP models out there all available with pre-trained weight on huge datasets. I could be overwhelming to decide which model to use. While this repo focuses on the Transformer family, here is come quick tips towards choosing your model.

BERT is certainly outperforming several NLP models that was previously state-of-the-art. Its performance improvement is largely attributed to its bidirectional transformer using Masked Language Model. RoBERTa, DistilBERT and XLNet are three powerful models that are popularly used and their various versions are available in `Hugging Face`. RoBERTa is a retraining of BERT with 1000% more training data and stronger compute power. In addition, dynamic masking is used during training. DistilBERT trains similarly to BERT but it has only half the number of parameters by using a technique called distillation. DistilBERT is not necessary more accurate then its counterparts but it requires less time to train and it’s faster during inference time. XLNet is the heavy weight player, it’s trained with larger datasets with much stronger computing power and longer time (about 5 times more than BERT). Moreover, during training time, XLNet doesn’t adopt masked language model, unlike BERT/RoBERTa/DistilBERT, but uses permutation language modeling where all tokens are predicted but in random order. The benefit of this is that the model could potentially learn the dependencies between all words. (with masked language model, dependencies between masked words are lost)

#### Conclusion

BERT/RoBERTa are very good baseline models are should perform fairly well for most NLP tasks. XLNet’s permutation based training could potentially give us a performance boost but the fine-tuning and inference  takes more time. If we want fast inference speed, go for DistilBERT.



# Transformer Key Elements

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

You will find two Jupyter notebooks in this repo. In the [Transformer with no LM fine-tuning.ipynb](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Transformer%20with%20no%20LM%20fine-tuning.ipynb) we will implement transformer with `FastAI` library without fine-tuning the language model separately. This notebook follows strictly the guide provided by [Maximilien Roberti](https://towardsdatascience.com/@maximilienroberti). Details of his guide can be found both in his [Medium post](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2) as well as [Kaggle chanllange](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta). (Thumbs up for Maximilien Roberti)

The reason why there exists a second notebook is because in Maximilien’s [implementation](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta) he mainly replaced the AWD_LSTM model from `FastAI` by **RoBERTa** from `Hugging Face`. According to Jeremy’s [ULMFiT paper](https://arxiv.org/pdf/1801.06146.pdf), if we’d fine tuning the encoder of the **RoBERTa** model as a language model before constructing the classifier for sentiment analysis task the accuracy could be even better. Hence, in the [Implement various Transformers with FastAI.ipynb]([https://github.com/Sylar257/Transformers-in-NLP/blob/master/Implement%20various%20Transformers%20with%20FastAI.ipynb](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Implement various Transformers with FastAI.ipynb)) you will find the complete code for building two separate `databunch` for both language model and classification task as well as how can we apply *transfer learning* with `transformers` while taking advantage of the convenience of `FastAI` toolkit.

To benchmark our result we will be using the classic [IMDb sentiment analysis dataset](https://s3.amazonaws.com/fast-ai-nlp/imdb). Following the standard ULMFiT approach, in the last [repo](https://github.com/Sylar257/ULMFiT-Sentiment-Analysis), we were able to reach **94.7%** accuracy which is slightly better than the state-of-the-art result in 2017 (94.1% accuracy). Now let’s challenge ourselves to push these results even further by implementing more advanced skills.

In this repository you will find everything you need to incorporate transformers as the base architecture when using the `FastAI` framework. I will also try to provide you with the essential explanations of why we would make those customizations so that when you choose a different architecture you can replicate the process.



### Transformer with no LM fine-tuning

When using the pre-trained models from `transformer` library, each model architecture need the following information:

1. A **model class** too lead/store a particular pre-trained model.
2. A **tokenizer class** to pre-process the data and make it compatible with our model of selection
3. A **configuration class** to load/store the configuration of a particular model

We will start with the [**RoBERTa** transformer](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) from Facebook. To incorporate customized model with `FastAI` is not as simple as plugging in the model architecture into the `Learner`. We have to construct the proper dataloader(in `FastAI` known as `databunch`) so that the transformer is getting what its expecting to get.

#### Step.1 Customize our processors:

The important thing here is that `FastAI` uses processors to perform repetitive tasks when creating DataBunch. A set of default processors are performed for [fastai.textlearners](https://github.com/fastai/fastai/blob/67308f15394bd8189eb9b3fbb3db770c6c78039e/fastai/text/data.py#L283). For example:

```python
# FastAI use various processors to perform repeatative tasks in data pipeline

def _get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, 
                              mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos),
            NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]
```

Here two default processors are invoked: tokenizer and numericalizer. The two classes are defined as such:

```python
class TokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, 
                 mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields
        self.include_bos, self.include_eos = include_bos, include_eos

    def process_one(self, item):
        return self.tokenizer._process_all_1(_join_texts([item], self.mark_fields, self.include_bos, self.include_eos))[0]

    def process(self, ds):
        ds.items = _join_texts(ds.items, self.mark_fields, self.include_bos, self.include_eos)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

class NumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=3):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)
```

This is our IMDb text after tokenized with `Fastai` default tokenizer:

![fastai_processor](images/fastai_processor.png)

and this is IMDb text tokenized with `transformer` `RoBERTa` tokenizer:

![transformer_processor](images/transformer_processor.png)

It’s easy to see that the `Hugging Face` `RoBERTa` is expecting different tokenization and that we need to customize the processors. What we need to do is essentially grab the tokenizer, vocabulary, special tokens and numericalizer from `Hugging Face` and put the `FastAI` wrapper on to make it a compatible class. Detail of these step please refer to the [Jupyter notebook](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Transformer%20with%20no%20LM%20fine-tuning.ipynb).

#### Step.2 Create Our `DataBunch` for classification:

We can easily find the IMDb dataset once we installed the `FastAI` library by calling `untar_data(URLs.IMDb)`. This would return us a `PosixPath` object under which path we can find three useful files: *train, test and unsup.*

In this notebook, we are going to ignore the *unsup* folder and construct `DataBunch` only using *train* and *test* folders. 

**\*Note that for `transformer_processor` we need to include `OpenFileProcessor()`. This is because we are reading from different folders instead of straight from dataframes or .csv files.**

```python
# processors need to be in the right order
transformer_processor = [OpenFileProcessor(),tokenize_processor, numericalize_processor]
```

Then create our `DataBunch`:

```python
# we can play around with the batch_size as long as the GPU can take it
bs = 16
data_clas = (TextList.from_folder(path, vocab=transformer_vocab, processor=transformer_processor)                       # specify the path
           .filter_by_folder(include=['train','test']) # exclude other folders
           .split_by_folder(valid='test')              # split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
           .label_from_folder(classes=['neg', 'pos'])  # label them all with their folders
           .databunch(bs=bs))                          # convert to databunch for the learner later
```

#### Step.3 Create our customized transformer model:

This step is quite straight forward. Get the model architecture itself provided by `Hugging Face` and make sure the `forward` function provides the right information. 

```python
# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        #attention_mask = (input_ids!=1).type(input_ids.type()) # Test attention_mask for RoBERTa
        
        logits = self.transformer(input_ids,
                                attention_mask = attention_mask)[0]   
        return logits

# hyper parameter setup
config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 2
    
    
# Create customized transformer model
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
```

Pay attention to what the `forward` function returns as we will be modifying this part once we get to the [next notebook](https://github.com/Sylar257/Transformers-in-NLP/blob/master/Implement%20various%20Transformers%20with%20FastAI.ipynb) and start to build language models. We should also get the *hyper-parameters* provided by `Hugging Face` by calling `config_class.from_pretrained(pretrained_model_name)`. Change the `config.num_label = 2` as we only have **positive** or **negative** in IMDb prediction and then we are good to go.

#### Step.4 Create Learner object:

A `DataBunch` and a `architecure` are two object we must provide in order to create a `Learner` object in `FastAI`. There is one more thing we consider specify here that is our optimizer. `FastAI` uses [AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw) by default and it was pointed out by Maximilien that for reproducing *BertAdam* specific behavior we have to set `correct_bias = False`. Hence we use the `partial( )` function:

```python
# customize AdamW
CustomAdamW = partial(AdamW, correct_bias=False)

# Create learner
learner = Learner(data_clas, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])

# Show graph of learner stats and metrics after each epoch.
learner.callbacks.append(ShowGraph(learner))
```

We are also going to split the layers. `FastAI` employs the concept of layer groups in that we divide our layers into a few groups so that later we **have the freedom to freeze/unfreeze their weight or applier different learning rates for different layer groups**.

```python
# For roberta-base
list_layers = [learner.model.transformer.roberta.embeddings,
              learner.model.transformer.roberta.encoder.layer[0],
              learner.model.transformer.roberta.encoder.layer[1],
              learner.model.transformer.roberta.encoder.layer[2],
              learner.model.transformer.roberta.encoder.layer[3],
              learner.model.transformer.roberta.encoder.layer[4],
              learner.model.transformer.roberta.encoder.layer[5],
              learner.model.transformer.roberta.encoder.layer[6],
              learner.model.transformer.roberta.encoder.layer[7],
              learner.model.transformer.roberta.encoder.layer[8],
              learner.model.transformer.roberta.encoder.layer[9],
              learner.model.transformer.roberta.encoder.layer[10],
              learner.model.transformer.roberta.encoder.layer[11],
              learner.model.transformer.roberta.pooler]

# use the defaul split function from Learner class
learner.split(list_layers)
```

#### Step.5 Training

Yeah! After all the hard work of setting things up, our favorite part of deep learning has come!

We are going to start with a frozen model by calling `learner.freeze()` , find the optimal learning rate `learner.lr_find()` and `learner.recorder.plot(skip_end = 10, suggestion=True)` and finally `learner.fit_one_cycle()` to do *learning rate annealing*.

![lr_find](images/lr_find.png)

As suggested by Jeremy Howard in his [ULMFiT paper](https://arxiv.org/pdf/1801.06146.pdf) we consider gradually unfreeze the layers 2 groups at a time, and repeat the above process.

We are able to achieve **95.3%** accuracy which higher than the** 94.1%**(2017 state-of-the-art) and 94.7% ULMFiT with AWD_LSTM bacs architecture.

![training_result_notebook_1](/home/projectx/Documents/Transformers-in-NLP/images/training_result_notebook_1.png)