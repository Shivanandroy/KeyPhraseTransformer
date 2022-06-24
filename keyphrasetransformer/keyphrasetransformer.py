# import
import os
import sys
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, T5ForConditionalGeneration

nltk.download('punkt')
nltk.download("words")

class KeyPhraseTransformer:
    def __init__(self, model_name: str = "snrspeaks/KeyPhraseTransformer"):
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def split_into_paragraphs(self, doc: str, max_tokens_per_para: int = 128):
        sentences = sent_tokenize(doc.strip())
        temp = ""
        temp_list = []
        final_list = []

        for i, sentence in enumerate(sentences):
            sent = sentence
            temp = temp + " " + sent
            wc_temp = len(self.tokenizer.tokenize(temp))

            if wc_temp < max_tokens_per_para:
                temp_list.append(sentence)

                if i == len(sentences) - 1:
                    final_list.append(" ".join(temp_list))

            else:
                final_list.append(" ".join(temp_list))

                temp = sentence
                temp_list = [sentence]

                if i == len(sentences) - 1:
                    final_list.append(" ".join(temp_list))

        return [para for para in final_list if len(para.strip()) != 0]

    def process_outputs(self, outputs):
        temp = [output[0].split(" | ") for output in outputs]
        flatten = [item for sublist in temp for item in sublist]
        return sorted(set(flatten), key=flatten.index)

    def filter_outputs(self, key_phrases, text):
        key_phrases = [elem.lower() for elem in key_phrases]
        text = text.lower()

        valid_phrases = []
        invalid_phrases = []

        for phrases in key_phrases:
            for phrase in word_tokenize(phrases):
                if (phrase in word_tokenize(text)) or (phrase in words.words()):
                    if phrases not in valid_phrases:
                        valid_phrases.append(phrases)
                else:
                    invalid_phrases.append(phrases)

        return [elem for elem in valid_phrases if elem not in invalid_phrases]

    def predict(self, doc: str):
        input_ids = self.tokenizer.encode(
            doc, return_tensors="pt", add_special_tokens=True
        )
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=2,
            max_length=512,
            repetition_penalty=2.5,
            length_penalty=1,
            early_stopping=True,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        return preds

    def get_key_phrases(self, text: str, text_block_size: int = 64):
        results = []
        paras = self.split_into_paragraphs(
            doc=text, max_tokens_per_para=text_block_size
        )

        for para in paras:
            results.append(self.predict(para))

        key_phrases = self.filter_outputs(self.process_outputs(results), text)
        return key_phrases
