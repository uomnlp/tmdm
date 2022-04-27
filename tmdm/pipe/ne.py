import itertools

from loguru import logger
from spacy.tokens import Doc
from typing import Iterable

from tmdm.classes import Provider


def doc_gen(stream, nes_batch):
    for doc, nes in zip(stream, nes_batch):
        doc._.nes = nes
        yield doc


class NEPipe:
    name = "ne-pipe"

    def __init__(self, vocab, provider: Provider):
        self.vocab = vocab
        self.provider = provider

    def __call__(self, doc: Doc):
        annotations = self.provider.annotate_document(doc)
        logger.info(annotations)
        doc._.nes = annotations
        return doc

    def pipe(self, stream: Iterable[Doc], batch_size: int):
        stream, copy = itertools.tee(stream)
        nes_batch = self.provider.annotate_batch(copy)
        return doc_gen(stream, nes_batch)
