# import
import os
import sys
import streamlit as st
import nltk
from nltk.corpus import words
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image

nltk.download("words")

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from keyphrasetransformer import KeyPhraseTransformer

logo = Image.open("_assets/logo.png")

# page config
st.set_page_config(layout="wide")

coll, colc, colr = st.columns([1, 2, 1])
with coll:
    st.write("")

with colc:
    st.image(logo)
    st.caption(
        "KeyPhraseTransformer lets you quickly extract key-phrases, topics or popular themes from your text data with the power of transformers"
    )
    st.caption(
        """
    KeyPhraseTransformer is built on mT5 Transformer architecture, trained on 
    500,000 training samples on 8 TPU cores. It is further fine-tuned on 50,000 data samples on general english corpus.

    """
    )

with colr:
    st.write("")


@st.cache(hash_funcs={KeyPhraseTransformer: lambda _: None}, allow_output_mutation=True)
def preparing_keyphraseformer():
    kp = KeyPhraseTransformer()
    return kp


kp = preparing_keyphraseformer()

# layout
col1, col2, col3 = st.columns([6, 1, 5])


# process
with col1:
    init_text = """Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code"""
    text = st.text_area(
        label="Text", value=init_text, placeholder="Paste your text here...", height=300
    )
    text_block_size = st.slider(
        label="Text block size", min_value=32, max_value=512, step=16, value=64
    )
    button = st.button("Extract")


with col3:
    if button:
        outputs = {
            "KeyPhraseTransformer": kp.get_key_phrases(
                text=text, text_block_size=text_block_size
            )
        }
        st.caption(
            """
        `T5-Base ● Epoch:3 ● Data:50,000 ● Time:60 Hrs (TPU)`
        """
        )

        st.json(outputs)
