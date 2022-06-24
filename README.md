<img align="center" src="_assets/logo1.png" alt="simpleT5">

<p align="center">
<b>Quickly extract key-phrases/topics from you text data with T5 transformer
</b>


**KeyPhraseTransformer** is built on T5 Transformer architecture, trained on 500,000 training samples to extract important phrases/topics/themes from text of any length.

### Why KeyPhraseTransformer?
- You get the power of amazing T5 architecture. 
- The underlying T5 model is specifically trained in extracting important phrases from the text corpus, so the results are of superior quality.
- No pre-processing is needed of any kind. Just dump your data to the model
- It does not need any n-gram-related inputs from user. It can automatically extract unigram, bigram, or trigram on its own.
- It can process text data of any length as it breaks down input text into smaller chunks internally
- It helps to automate the topic modeling/keyword extraction process end to end with no manual intervention.

### Installation:
```python
pip install keyphrasetransformer
```
### Use:
[![Generic badge](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/snrspeaks/KeyPhraseTransformer)

```python
from keyphrasetransformer import KeyPhraseTransformer

kp = KeyPhraseTransformer()

doc = """
Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned 
on a downstream task, has emerged as a powerful technique in natural language processing (NLP). 
The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. 
In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework 
that converts every language problem into a text-to-text format. Our systematic study compares pretraining objectives, 
architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. 
By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, 
we achieve state-of-the-art results on many benchmarks covering summarization, question answering, 
text classification, and more. To facilitate future work on transfer learning for NLP, 
we release our dataset, pre-trained models, and code.

"""

kp.get_key_phrases(doc)
```
```
['transfer learning',
 'natural language processing (nlp)',
 'nlp',
 'text-to-text',
 'language understanding',
 'transfer approach',
 'pretraining objectives',
 'corpus',
 'summarization',
 'question answering']
 ```