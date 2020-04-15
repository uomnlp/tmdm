from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, List, Iterable, Union

from spacy.tokens import Doc

ERTuple = namedtuple("ERTuple", ["entities", "relations"])

# Typed Entities and Relationships
EntitiesAnnotation = List[Tuple[int, int, str]]
EntitiesRelationshipsAnnotation = Tuple[EntitiesAnnotation, List[Tuple[int, int, str]]]
OffsetAnnotation = Union[EntitiesAnnotation, EntitiesRelationshipsAnnotation]


class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...

    @abstractmethod
    def annotate_document(self, doc: Doc) -> OffsetAnnotation:
        ...

    def annotate_batch(self, docs: List[Doc]) -> List[OffsetAnnotation]:
        # TODO if something fails, need to convert to list
        return [self.annotate_document(doc) for doc in docs]
