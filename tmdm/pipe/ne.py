from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc
from typing import Iterable, List

from tmdm.classes import Provider



class NEPipe:
    name = "ne-pipe"

    def __init__(self, provider: Provider):
        self.provider = provider

    def __call__(self, doc: Doc):
        annotations = self.provider.annotate_document(doc)
        logger.debug(annotations)
        doc._.nes = annotations
        return doc


    def pipe(self, docs: List[Doc]):
        annotated_batch = self.provider.annotate_batch(docs)
        assert len(docs) == len(annotated_batch)
        for doc, annotations in zip(docs, annotated_batch):
            doc._.nes = annotations
        return docs


class ELPipe(NEPipe):
    name = 'el-pipe'
