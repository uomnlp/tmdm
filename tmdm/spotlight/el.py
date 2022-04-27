from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc, Span

from tmdm.classes import CharOffsetAnnotation, Provider
from tmdm.pipe.pipe import PipeElement
from tmdm.util import get_offsets_from_sentences
from spotlight import annotate as spotlight_annotate
from spotlight import SpotlightException
from requests.exceptions import HTTPError

SpotlightDocAnnotation = List[Dict[str, Any]]


def annotate(*args, **kwargs):
    try:
        return spotlight_annotate(*args, **kwargs)
    except SpotlightException:
        return []
    except HTTPError as e:
        if '400' in str(e):
            return []
        else:
            raise e


def select_meta(ent, rich):
    if rich:
        return {'uri': ent['URI'], 'support': ent['support'], 'types': ent['types'].split(',') if ent['types'] else [],
                'similarity': ent['similarityScore'], 'label': ent['URI']}
    else:
        return ent['URI']


def find_within(ents, start, end):
    for e in ents:
        logger.debug(f"({start},{end})  vs ({e['offset']},{e['offset'] + len(e['surfaceForm'])})")
        if e['offset'] >= start and e['offset'] + len(e['surfaceForm']) <= end:
            yield e


def convert(doc: Doc, result: SpotlightDocAnnotation, rich, nes_only, threshold) -> CharOffsetAnnotation:
    results = []
    filtered_ents = [r for r in result if r['similarityScore'] > threshold]
    if nes_only:
        for ne, (s, e, l) in zip(doc._.nes, doc._._nes):
            # find all ents that are within ne
            # select the one with the highest mention score? the longest one? dunno
            ents = list(find_within(filtered_ents, s, e))
            logger.debug(ents)
            if ents:
                ent = max(ents, key=lambda e: len(e['surfaceForm']))
                t = select_meta(ent, rich)

                ## TODO: enough overlap?
                logger.debug(f"Mapping {ne} to {t['uri']}.")
                if rich:
                    t['label'] = t['label'] = ne.label_
                results.append((s, e, t))
            else:
                results.append((s, e, l))
    else:
        for ent in filtered_ents:
            logger.debug(ent)
            t = select_meta(ent, rich)
            s = ent['offset']
            e = s + len(ent['surfaceForm'])

            results.append((s, e, t))
    return results


class OnlineELProvider(Provider):
    name = 'spotlight-el-provider'

    def __init__(self, endpoint, types=None, rich=False, nes_only=False, threshold=0.6):
        self.endpoint = endpoint
        self.types = types
        self.rich = rich
        self.nes_only = nes_only
        self.threshold = threshold

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def annotate_document(self, doc: Doc) -> CharOffsetAnnotation:
        return self.annotate_batch([doc])[0]

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        docs = list(docs)
        # try:

        result = [annotate(self.endpoint, str(i)) for i in docs]
        logger.debug(f"Result: {result}")
        assert len(result) == len(docs)

        return [convert(doc, r, self.rich, self.nes_only, threshold=self.threshold) for doc, r in zip(docs, result)]


def get_el_pipe(model: str = None, endpoint='http://kant.cs.man.ac.uk:2222/rest/annotate', rich=True, nes_only=False, threshold=0.9):
    return PipeElement(name='el', field='nes', provider=OnlineELProvider(endpoint, rich=rich, nes_only=nes_only, threshold=threshold))
