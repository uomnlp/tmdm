from flair.models import SequenceTagger
from flair.data import Sentence
import torch

import unicodedata

import numpy as np
import seqeval.metrics
import spacy
import torch
from tqdm import tqdm, trange
# from transformers import LukeTokenizer, LukeForEntitySpanClassification


def get_model(parameters=None):
    return Flair(parameters)


class NER_model:
    def __init__(self, parameters=None):
        pass

    def predict(self, sents):
        """Sents: List of plain text consequtive sentences. 
        Returns a dictionary consisting of a list of sentences and a list of mentions, where for each mention AT LEAST (it may give additional information) the following information is given:
            sent_idx - the index of the sentence that contains the mention
            text - the textual span that we hypothesise that represents an entity
            start_pos - the character idx at which the textual mention starts 
            end_pos - the character idx at which the mention ends"""
        pass

# class Flair(NER_model):
#     def __init__(self, parameters=None):
#         self.model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003").eval().to("cuda")
#         self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
#         self.nlp = spacy.load("en_core_web_sm")

#     def predict(self, sentences):
#         text = sentences[0]
#         spans = []
#         cur = 0
# #         for text in txt.split("."):
# #             if not text: continue
# #             text += "."
#         doc = self.nlp(text)
#         entity_spans = []
#         original_word_spans = []
#         i2idx = {}
#         for token_start in doc:
#             if token_start.text.strip():
#                 for token_end in doc[token_start.i:token_start.i+8]:
#                     entity_spans.append((token_start.idx, token_end.idx + len(token_end)))
#                     original_word_spans.append((token_start.i, token_end.i + 1))
#                     i2idx[(token_start.i, token_end.i + 1)] = (token_start.idx, token_end.idx + len(token_end))

#         inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt", padding=True)
#         inputs = inputs.to("cuda")
#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         logits = outputs.logits
#         max_logits, max_indices = logits[0].max(dim=1)
#         if sum(max_indices) == 0: print("overflow")
#         predictions = []
#         for logit, index, span in zip(max_logits, max_indices, original_word_spans):
#             if index != 0:  # the span is not NIL
#                   predictions.append((logit, span, self.model.config.id2label[int(index)]))

#         # # construct an IOB2 label sequence
#         predicted_sequence = ["O"] * len(doc)
#         for score, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
#           if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
#                 predicted_sequence[span[0]] = "B-" + label
#                 offset = i2idx[(span[0],span[1])]
#                 spans.append({
#                   "text": f'{text[offset[0]:offset[1]]}',
#                   'start_pos': cur+offset[0], 
#                   'end_pos': cur+offset[1], 
#                   'labels': [f"{label} ({score})"], 
#                   'sent_idx': 0,
#                 }
#                   )
#                 if span[1] - span[0] > 1:
#                     predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)
#         cur += len(text)
#         for s in spans:
#             assert s["text"] == text[s["start_pos"]:s["end_pos"]], s
#         return {"sentences": sentences, "mentions": spans}
    
    
class Flair(NER_model):
    def __init__(self, parameters=None):
        self.model = SequenceTagger.load("ner")
#         self.model = SequenceTagger._init_model_with_state_dict(torch.load("ner-english-large/pytorch_model.bin",map_location='cuda:0'))

    def predict(self, sentences):
        mentions = []
        for sent_idx, sent in enumerate(sentences):
#             sent = Sentence(sent, use_tokenizer=True)
            sent = Sentence(sent)
            self.model.predict(sent)
            sent_mentions = sent.to_dict(tag_type="ner")["entities"]
            for mention in sent_mentions:
                mention["sent_idx"] = sent_idx
            mentions.extend(sent_mentions)
        return {"sentences": sentences, "mentions": mentions}

