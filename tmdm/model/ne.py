from typing import List, Optional
from spacy.tokens import Doc, Token, Span
from loguru import logger

from tmdm.model.extensions import Annotation, extend


@extend(Token, type='property', create_attribute=True, default=[])
def nes(self: Token):
    """


    Args:
        self:
        annotation_type:

    Returns:

    """
    logger.debug(f"Retrieving ne annotations for token '{self}'")
    logger.debug(f"Result is: '{self._._nes}'")
    return [
        NamedEntity.make(self.doc, i) for i in self._._nes
    ]


def set_nes(self: Doc, nes):
    # TODO if re-setting the NES, need to unload the previous token annotations
    # normalise and align, should be either in a pipeline element or a util function or something like that
    if self._._nes:
        # logger.error("Cannot re-set tokens (yet)!")
        # raise NotImplementedError("Cannot re-set tokens (yet)!")
        self._._nes = []
    for idx, (start, end, label) in enumerate(nes):
        logger.trace(nes)
        for token in self._.char_span_relaxed(start, end):
            annotations = token._._nes
            if idx not in annotations:
                annotations.append(idx)

    self._._nes = nes


Span.set_extension('ne_meta', default=None, force=True)


@extend(Doc, 'property', create_attribute=True, default=[], setter=set_nes)
def nes(self: Doc) -> List['NamedEntity']:
    """

    Args:
        self:
        tags:
        ner_generator:

    Returns:

    """
    tags = self._._nes
    nes = []
    if not tags:
        logger.warning("No NEs extracted for this document (yet?).")

    for i, _ in enumerate(tags):
        nes.append(NamedEntity.make(self.doc, i))
    return nes


@extend(Span)
def get_ne(self: Span) -> Optional['NamedEntity']:
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    return next(
        (NamedEntity.make(self.doc, i) for i, (s, e, _) in enumerate(self.doc._._nes) if start == s and end == e),
        None
    )


@extend(Span)
def is_ne(self: Span) -> bool:
    return self._.get_ne() is not None


@extend(Span)
def get_nes(self: Span):
    start = self[0].idx
    end = self[-1].idx + len(self[-1])
    logger.trace(f"start,end: {start},{end}")
    return [
        NamedEntity.make(self.doc, i) for i, (s, e, _) in enumerate(self.doc._._nes) if start <= s and end >= e
    ]


@extend(Span)
def has_nes(self: Span):
    return not self._.get_nes() == []


class NamedEntity(Annotation):

    def identical(self, other: 'NamedEntity'):
        return isinstance(other, NamedEntity) and self.kb_id == other.kb_id

    @classmethod
    def make(cls, doc: Doc, idx, *args, **kwargs) -> 'NamedEntity':
        start, end, label = doc._._nes[idx]
        if isinstance(label, str):
            span = super().make(doc, idx, start, end, label, kb_id=f"{label}/{str(doc._.char_span_relaxed(start, end))}")
            span._.ne_meta = {}
            return span
        elif isinstance(label, dict):
            span = super().make(doc, idx, start, end, label=label['label'])
            span._.ne_meta = label
            return span
