from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc
from dateutil.parser import parse

from tmdm.classes import CharOffsetAnnotation
from tmdm.pipe.pipe import PipeElement
from tmdm.transformers.common import OnlineProvider
from tmdm.util import get_offsets_from_sentences
from transformers.pipelines import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

DocumentLevelTransformerEntitiesAnnotation = List[List[Dict[str, Any]]]


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


def convert(doc: Doc, result: DocumentLevelTransformerEntitiesAnnotation, filter_subwords=True) -> CharOffsetAnnotation:
    results = []
    assert len(list(doc.sents)) == len(result)

    for sent, ann in zip(doc.sents, result):
        sent_offset = sent.start_char
        for s, e, t, w in [(e['start'], e['end'], e['entity_group'], e['word']) for e in ann]:
            if filter_subwords and w.startswith('##'):
                pass
            else:
                results.append((sent_offset + s, sent_offset + e, t))
    return results


class OnlineNerProvider(OnlineProvider):
    def __init__(self, task: str, with_date, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        if with_date:
            self.date_model = SequenceTagger.load("flair/ner-english-ontonotes-large")
        else:
            self.date_model = None

    #         if with_date: self.date_model = SequenceTagger._init_model_with_state_dict(torch.load("./BLINK/ner-english-ontonotes-large/pytorch_model.bin", map_location='cuda:0'))
    #         else: self.date_model = None

    def load(self, _=None):
        self.pipeline = pipeline(self.task, model=self.path_or_name, tokenizer=self.path_or_name_tokenizer,
                                 device=self.cuda, aggregation_strategy='simple')

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        docs = list(docs)
        instances = [self.preprocess(doc) for doc in docs] if self.preprocess else docs
        # try:
        flat_instances = [i for l in instances for i in l]
        if not flat_instances:
            logger.debug("Everything is empty!")
            return [[] for _ in docs]
        logger.debug(flat_instances)
        result = self.pipeline(flat_instances)
        if not result:
            logger.debug("No named entities recognized!")
            return [[] for _ in docs]
        if result and not isinstance(result[0], list):
            result = [result]
        logger.debug(f"Result: {result}")
        # except Exception as e:
        #    logger.error(str(e))
        #    return [([], []) for _ in docs]
        result_iterator = iter(result)
        # batched_results = [[self.pipeline.group_entities(next(result_iterator)) for _ in d.sents] for d in docs]
        batched_results = [[next(result_iterator) for _ in d.sents] for d in docs]
        logger.debug(batched_results)
        batched_results_iter = iter(batched_results)
        ret = [self.converter(doc, next(batched_results_iter)) for doc in docs]
        mentions = []
        if self.date_model:
            for idx, doc in enumerate(docs):
                sent = Sentence(str(doc))
                self.date_model.predict(sent)
                sent_mentions = sent.to_dict(tag_type="ner")["entities"]
                for mention in sent_mentions:
                    label = str(mention["labels"][0]).split()[0]
                    m = mention["text"].lower()
                    if label == "DATE" and is_date(m) and len(m) > 2:
                        ret[idx].append((mention["start_pos"], mention["end_pos"], label))
        #                         mentions.append(mention)
        return ret


def get_ne_pipe(model: str = None, tokenizer: str = None, cuda=-1, with_date=False):
    return PipeElement(name='ner', field='nes',
                       provider=OnlineNerProvider(task="ner", path_or_name=model,
                                                  path_or_name_tokenizer=tokenizer,
                                                  converter=convert, cuda=cuda, with_date=with_date))