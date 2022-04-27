from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc

from tmdm.classes import CharOffsetAnnotation
from tmdm.transformers.common import OnlineProvider
from tmdm.util import get_offsets_from_sentences
from spotlight import annotate

SpotlightDocAnnotation = List[Dict[str, Any]]


def convert(doc: Doc, result: SpotlightDocAnnotation) -> CharOffsetAnnotation:
    results = []
    assert len(list(doc.sents)) == len(result)
    for ent in result:
        t = ent['URI']
        s = ent['offset']
        e = s + len(ent['surface_form'])
        results.append((s, e, t))
    return results


class OnlineNerProvider(OnlineProvider):
    def __init__(self, task: str, endpoint, types=None,):
        self.endpoint = endpoint
        self.types = types
        super().__init__(task)

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        docs = list(docs)
        instances = [self.preprocess(doc) for doc in docs] if self.preprocess else docs
        # try:


        result = [annotate(self.endpoint, i) for i in instances]
        logger.debug(f"Result: {result}")
        assert len(result) == len(docs)

        return [convert(doc, r) for doc, r in zip(docs, result)]


def get_ne_provider(model: str = None, endpoint='http://kant.cs.man.ac.uk:2222'):
    return OnlineNerProvider("ner", endpoint)
