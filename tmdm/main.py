import spacy
from typing import Tuple, Any, Union, Callable, Optional
from tmdm.pipe.pipe import Pipeline
# from loguru import logger
# from scispacy.custom_tokenizer import combined_rule_tokenizer
# from spacy.language import Language

# from tmdm.classes import Provider
# from tmdm.cached import Cached
# from tmdm.pipe.coref import CorefPipe

#from tmdm.pipe.tokenizer import IDAssigner
# from tmdm.util import failsafe_combined_rule_sentence_segmenter, OneSentSentencizer


def change_getter(nlp, getter=Callable[[Any, ], Tuple[str, str]]):
    nlp.tokenizer.getter = getter


def default_getter(d):
    return d['id'], d['abstract']


def default_one_sent_getter(d):
    uid, data = d
    return uid, data


def tmdm_pipeline(getter: Optional[Callable[[Any, ], Tuple[str, str]]] = None, model='en_core_web_sm',
                  disable=None, with_ids=False) -> Pipeline:
    disable = disable or ['ner', 'parse']
    nlp = spacy.load(model, disable=disable)
    # if not with_ids:
    #
    #     nlp.tokenizer = IDAssigner(nlp.Defaults.create_tokenizer(nlp), getter, generate_ids=True)
    # else:
    #     nlp.tokenizer = IDAssigner(nlp.Defaults.create_tokenizer(nlp), getter, generate_ids=False)
    #     change_getter(nlp, getter)
    if not getter and with_ids:
        raise ValueError("Data comes with ids, but no getter configured!")
    return Pipeline(nlp, pipes=None, getter=getter, generate_ids=not with_ids)


# def tmdm_scientific_pipeline(getter: Callable[[Any, ], Tuple[str, str]] = default_getter, model="en_core_sci_lg"):
#     nlp = spacy.load(model, disable=['ner', 'parser'])
#     nlp.tokenizer = IDAssigner(combined_rule_tokenizer(nlp), getter=getter)
#     nlp.add_pipe(failsafe_combined_rule_sentence_segmenter)
#     return nlp
#
#
# def tmdm_one_sent_pipeline(getter: Callable[[Any, ], Tuple[str, str]] = default_one_sent_getter,
#                            model="en_core_sci_lg"):
#     nlp = spacy.load(model, disable=['ner', 'parser'])
#     nlp.tokenizer = IDAssigner(combined_rule_tokenizer(nlp), getter=getter)
#     nlp.add_pipe(OneSentSentencizer())
#     return nlp

#
# def add_ner(nlp, provider: Union[Provider, str], schema="list_of_tuples_bio_stacked"):
#     if isinstance(provider, str):
#         provider = Cached(path=provider, getter=lambda d: d['abstract'], schema=schema)
#     nlp.add_pipe(PipeElement())
#
#
# def add_el(nlp, provider: Union[Provider, str], schema="list_of_tuples_bio_stacked"):
#     if isinstance(provider, str):
#         provider = Cached(path=provider, getter=lambda d: d['abstract'], schema=schema)
#     nlp.add_pipe(ELPipe(nlp.vocab, provider))
#
#
# def add_oie(nlp, provider: Union[Provider, str], schema='list_of_tuples_bio_stacked'):
#     if isinstance(provider, str):
#         provider = Cached(path=provider, getter=lambda d: d['abstract'], schema=schema)
#     nlp.add_pipe(OIEPipe(nlp.vocab, provider))
#
#
# def add_coref(nlp, provider: Union[Provider, str], schema='list_of_tuples_bio_stacked'):
#     if isinstance(provider, str):
#         provider = Cached(path=provider, getter=lambda d: d['abstract'], schema=schema)
#     nlp.add_pipe(CorefPipe(nlp.vocab, provider))


# __all__ = [
#     x.__name__ for x in (tmdm_pipeline)
# ]
