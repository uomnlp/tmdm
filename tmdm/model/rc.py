from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Dict, Iterable, Optional

from fastcache import clru_cache
from loguru import logger
from spacy.tokens import Token, Doc, Span

from tmdm.classes import ERTuple
from tmdm.model.extensions import Annotation, extend


def set_relations(self: Doc, relations):
    if self._._relations:
        logger.error("Cannot re-set tokens (yet)!")
        raise NotImplementedError("Cannot re-set tokens (yet)!")

    self._._relations = relations


@extend(Doc, 'property', create_attribute=True, default=[], setter=set_relations)
def relations(self: Doc):
    if not self._._relations:
        logger.warning(f"No relations extracted for this document (yet?): {self._._relations}.")

    Relation.doc = self.doc
    # Two tuples in _relations for each Relation instance in relations
    return [Relation.make(ind) for ind in range(0, len(self._._relations), 2)]


class Relation:
    doc: Doc

    def __init__(self, idx, relation_type):
        self.idx = idx
        self.relation_type = relation_type

    @property
    def subject(self):
        ind = self.idx * 2
        (start, end, label) = self.doc._._relations[ind]
        return Annotation.make(self.doc, ind, start, end, label)

    @property
    def object(self):
        ind = self.idx * 2
        (start, end, label) = self.doc._._relations[ind + 1]
        return Annotation.make(self.doc, ind + 1, start, end, label)

    @classmethod
    def make(cls, ind: int):
        # Subject and object at ind and ind+1 where ind = 2*idx
        (_, _, subjlabel) = Relation.doc._._relations[ind]
        relation_type = subjlabel.split("-")[-1]
        relation = Relation(int(ind / 2), relation_type)
        logger.debug(f"{relation.subject} - {relation.relation_type} - {relation.object}")
        return relation
