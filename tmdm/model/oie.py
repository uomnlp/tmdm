from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Dict, Iterable, Optional

from fastcache import clru_cache
from loguru import logger
from spacy.tokens import Token, Doc, Span

from tmdm.classes import ERTuple
from tmdm.model.extensions import Annotation, extend


# HAS SCIENCE GONE TOO FAR?
def _make(doc, i):
    return (Verb if doc._._oies.entities[i][2] == "VERB" else Argument).make(doc, i)


@extend(Token, type='property', create_attribute=True, default=[])
def oies(self: Token):
    """


    Args:
        self:
        annotation_type:

    Returns:

    """
    logger.debug(f"Retrieving ne annotations for token '{self}'")
    logger.debug(f"Result is: '{self._._oies}'")
    return [
        _make(self.doc, i) for i in self._._oies
    ]


def set_oies(self: Doc, oies):
    tags, relations = oies
    if self._._oies:
        logger.error("Cannot re-set tokens (yet)!")
        raise NotImplementedError("Cannot re-set tokens (yet)!")

    for idx, (start, end, label) in enumerate(tags):
        logger.trace(f"oies:{tags}")
        for token in self._.char_span_relaxed(start, end):
            annotations = token._._oies
            if idx not in annotations:
                annotations.append(idx)

    self._._oies = ERTuple(*oies)


@extend(Doc, 'property', create_attribute=True, default=[], setter=set_oies)
def oies(self: Doc) -> List['Verb']:
    tags = self._._oies[0]
    if not tags:
        logger.warning("No NEs extracted for this document (yet?).")

    return [Verb.make(self.doc, i) for i, (_, _, label) in enumerate(tags) if label == 'VERB']


@clru_cache(maxsize=100000)
def _v2a(self: Doc, idx) -> Iterable[int]:
    arg_ids = [(arg_idx, label) for verb_idx, arg_idx, label in self._._oies.relations if verb_idx == idx]
    key = itemgetter(1)
    return (arg_idx for arg_idx, _ in sorted(arg_ids, key=key))


@clru_cache(maxsize=100000)
def _a2v(self: Doc, idx) -> Iterable[int]:
    return [verb_idx for verb_idx, arg_idx, _ in self._._oies.relations if arg_idx == idx]


class Verb(Annotation):
    _arguments: List[int]
    _continuous: bool = None
    _extended_span: Span = None

    @property
    def continuous(self) -> bool:
        if self._continuous is None:
            all_spans = sorted([(self.start, self.end)] + [(a.start, a.end) for a in self.arguments])
            # explosion of pythonicism
            continuous = all(
                end_previous == start_next for (_, end_previous), (start_next, _) in zip(all_spans, all_spans[1:])
            )
            self._continuous = continuous
            if self._continuous:
                self._extended_span = self.doc[slice(all_spans[0][0], all_spans[-1][1])]
        return self._continuous

    @property
    def extended_span(self) -> Optional[Span]:
        if self.continuous:
            return self._extended_span

    # _arguments_as_map: Dict[int, 'Argument']
    # extended_span: Tuple[int, int]
    # continuous: bool

    # def set_arguments(self, arguments: List['Argument']):
    #     self.arguments = arguments
    #     self.arguments_as_map = {a.label_: a for a in arguments}
    #
    def argument(self, order):
        return Argument.make(self.doc, self._arguments[order])

    @property
    def arguments(self) -> List['Argument']:
        return [Argument.make(self.doc, idx) for idx in self._arguments]

    @classmethod
    def make(cls, doc: Doc, idx: int, *args, **kwargs):
        start, end, label = doc._._oies.entities[idx]
        assert label == "VERB"
        verb = super().make(doc, idx, start, end, label)
        verb._arguments = _v2a(doc, idx)
        logger.debug(f"{verb} has arguments: {verb._arguments}")
        return verb


# Prune same arguments
# Argument.merge_all(arg for p in oie_extractions for arg in p.arguments)
# return oie_extractions

# result = []
# tags: List[List[Tuple[int, int, str]]] = self._._oies
# if not tags:
#     logger.warning("No OIEs extracted for this document (yet?).")
#     logger.debug(tags)
# for i, verb_tags in enumerate(tags):
#     logger.debug("Beginning...")
#     verb_start, verb_end = next((s, e) for s, e, l in verb_tags if l == "VERB")
#     span = self.char_span(verb_start, verb_end)
#     verb_start, verb_end = span.start, span.end
#     verb = Verb(self, verb_start, verb_end, "VERB")
#     logger.info(f"Processing: {verb}")
#     verb.update_token_annotations(i)
#     args = []
#     logger.debug(f"Processed {verb}")
#     for j, (arg_start, arg_end, order) in enumerate(((s, e, l) for s, e, l in verb_tags if not l == 'VERB'),
#                                                     len(Argument.cache[tmdm.model.extensions.name])):
#         # TODO: potentially: split order by '-'
#         span = self.char_span(arg_start, arg_end)
#         arg_start, arg_end = span.start, span.end
#         a = Argument(self, arg_start, arg_end, order)
#         a.update_token_annotations(j)
#         a.verb = verb
#         args.append(a)
#
#     verb.set_arguments(args)
#     all_spans = sorted([(verb_start, verb_end)] + [(a.start, a.end) for a in args])
#
#     # explosion of pythonicism
#     verb.continuous = all(
#         end_previous == start_next for (_, end_previous), (start_next, _)
#         in zip(all_spans, all_spans[1:])
#     )
#     verb.extended_span = all_spans[0][0], all_spans[-1][1]
# return result

class Argument(Annotation):
    _verbs: List[int]

    @property
    def verbs(self) -> List[Verb]:
        return [Verb.make(self.doc, idx) for idx in self._verbs]

    @classmethod
    def make(cls, doc: Doc, idx: int, *args, **kwargs):
        verb_ids = _a2v(doc, idx)
        start, end, label = doc._._oies.entities[idx]
        assert label == "ARG"
        argument = super().make(doc, idx, start, end, label)
        argument._verbs = verb_ids
        return argument
