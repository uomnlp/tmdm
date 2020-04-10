from collections import defaultdict
from typing import List
from dynaconf import settings
from spacy.tokens import Doc, Token
from loguru import logger

from tmdm.model.extensions import Annotation, extend

force = settings['force']


# ATTRIBUTES
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
        NamedEntity.make(i, self.doc) for i in self._._nes
    ]


def set_nes(self, nes):
    # TODO if re-setting the NES, need to unload the previous token annotations
    # normalise and align, should be either in a pipeline element or a util function or something like that
    if self._._nes:
        logger.error("Cannot re-set tokens (yet)!")
        raise NotImplementedError("Cannot re-set tokens (yet)!")
    for idx, (start, end, label) in enumerate(nes):
        logger.debug(nes)
        for token in self._.char_span_relaxed(start, end):
            annotations = token._._nes
            if idx not in annotations:
                annotations.append(idx)

    self._._nes = nes


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


class NamedEntity(Annotation):
    cache = defaultdict(dict)

    def identical(self, other: 'NamedEntity'):
        return isinstance(other, NamedEntity) and self.kb_id == other.kb_id

    @classmethod
    def make(cls, doc: Doc, idx, *args, **kwargs) -> 'NamedEntity':
        start, end, label = doc._._nes[idx]
        return super().make(doc, idx, start, end, label, f"{label}/{str(doc.text[start:end])}")
