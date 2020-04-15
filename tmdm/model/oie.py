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
    return (Verb if doc._._oies.entities[i][2].startswith("V") else Argument).make(doc, i)


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

    return [Verb.make(self.doc, i) for i, (_, _, label) in enumerate(tags) if label == 'VERB' or label == "V"]


@clru_cache(maxsize=100000)
def _v2a(self: Doc, idx) -> Iterable[int]:
    arg_ids = [(arg_idx, label) for verb_idx, arg_idx, label in self._._oies.relations if verb_idx == idx]
    key = itemgetter(0)
    return [arg_idx for arg_idx, _ in sorted(arg_ids, key=key)]


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
            logger.trace(f"all spans: {all_spans}")
            # explosion of pythonicism
            continuous = all(
                end_previous == start_next for (_, end_previous), (start_next, _) in zip(all_spans, all_spans[1:])
            )
            self._continuous = continuous
            logger.trace(f"continuous?: {continuous}")
            if self._continuous:
                self._extended_span = self.doc[slice(all_spans[0][0], all_spans[-1][1])]
        return self._continuous

    @property
    def full_text(self):
        if self.arguments:
            span = [self.arguments[0]] + [self] + self.arguments[1:]
        else:
            span = [self]
        return " ".join(t.text for t in span)

    @property
    def extended_span(self) -> Optional[Span]:
        if self.continuous:
            return self._extended_span

    def argument(self, order):
        return Argument.make(self.doc, self._arguments[order])

    @property
    def core_arguments(self):
        return [arg for arg in self.arguments if "M" not in arg.order]

    @property
    def arguments(self) -> List['Argument']:
        return [Argument.make(self.doc, idx) for idx in self._arguments]

    @classmethod
    def make(cls, doc: Doc, idx: int, *args, **kwargs):
        start, end, label = doc._._oies.entities[idx]
        assert label == "VERB" or label == 'V'
        verb = super().make(doc, idx, start, end, label)
        verb._arguments = _v2a(doc, idx)
        logger.debug(f"{verb} has arguments: {verb._arguments}")
        return verb


@extend(Span)
def get_verb(self: Span) -> Optional['Verb']:
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    return next(
        (Verb.make(self.doc, i) for i, (s, e, l) in enumerate(self.doc._._oies.entities)
         if start == s and end == e and l.startswith("V")),
        None
    )


@extend(Span)
def is_verb(self: Span) -> bool:
    return self._.get_verb() is not None


@extend(Span)
def get_argument(self: Span) -> Optional['Argument']:
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    return next(
        (Argument.make(self.doc, i) for i, (s, e, l) in enumerate(self.doc._._oies.entities)
         if start == s and end == e and l.startswith("ARG")),
        None
    )


@extend(Span)
def is_argument(self: Span) -> bool:
    return self._.get_argument() is not None


@extend(Span)
def get_verbs(self: Span):
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    logger.trace(f"start,end: {start},{end}")
    return [
        Verb.make(self.doc, i) for i, (s, e, l) in enumerate(self.doc._._oies.entities)
        if start <= s and end >= e and l.startswith("V")
    ]


@extend(Span)
def has_verbs(self: Span):
    return not self._.get_verbs() == []


@extend(Span)
def get_arguments(self: Span) -> List['Argument']:
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    return [
        Argument.make(self.doc, i) for i, (s, e, l) in enumerate(self.doc._._oies.entities)
        if start <= s and end >= e and l.startswith("ARG")
    ]


@extend(Span)
def has_arguments(self: Span) -> bool:
    return not self._.get_arguments() == []


class Argument(Annotation):
    _verbs: List[int]
    order: str

    @property
    def verbs(self) -> List[Verb]:
        return [Verb.make(self.doc, idx) for idx in self._verbs]

    @classmethod
    def make(cls, doc: Doc, idx: int, *args, **kwargs):
        verb_ids = _a2v(doc, idx)
        start, end, label = doc._._oies.entities[idx]
        assert "ARG" in label
        argument = super().make(doc, idx, start, end, label)
        argument._verbs = verb_ids
        argument.order = label
        return argument
