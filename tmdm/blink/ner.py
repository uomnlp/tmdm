from flair.models import SequenceTagger
from flair.data import Sentence
from dateutil.parser import parse
import torch

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def get_model(parameters=None, with_date=False):
    return Flair(parameters, with_date)


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
    
    
class Flair(NER_model):
    def __init__(self, parameters=None, with_date=False):
        self.model = SequenceTagger.load("ner")
        if with_date: self.date_model = SequenceTagger.load("flair/ner-english-ontonotes-large")
        else: self.date_model = None
#         self.model = SequenceTagger._init_model_with_state_dict(torch.load("./BLINK/ner-english-large/pytorch_model.bin",map_location='cuda:0'))
#         if with_date: self.date_model = SequenceTagger._init_model_with_state_dict(torch.load("./BLINK/ner-english-ontonotes-large/pytorch_model.bin",map_location='cuda:0'))
#         else: self.date_model = None

    def predict(self, sentences):
        mentions = []
        for sent_idx, s in enumerate(sentences):
            sent = Sentence(s)
            self.model.predict(sent)
            sent_mentions = sent.to_dict(tag_type="ner")["entities"]
            for mention in sent_mentions:
                mention["sent_idx"] = sent_idx
            mentions.extend(sent_mentions)
            
            if self.date_model:
                sent_wdate = Sentence(s)
                self.date_model.predict(sent_wdate)
                sent_mentions_wdate = sent_wdate.to_dict(tag_type="ner")["entities"]
                for mention in sent_mentions_wdate:
                    mention["sent_idx"] = sent_idx
                    label = str(mention["labels"][0]).split()[0]
                    m = mention["text"].lower()
                    if label == "DATE" and is_date(m) and len(m) > 2:
                        mentions.append(mention)
        return {"sentences": sentences, "mentions": mentions}

