import itertools
from spacy.tokens import Doc
from typing import Iterable

from tmdm.classes import Provider


def oies_gen(stream, nes_batch):
    for doc, oies in zip(stream, nes_batch):
        doc._.oies = oies
        yield doc


class OIEPipe:
    name = "oie-pipe"

    def __init__(self, vocab, provider: Provider):
        self.vocab = vocab
        self.provider = provider

    def __call__(self, doc: Doc):
        annotations = self.provider.annotate_document(doc)
        doc._.oies = annotations
        return doc

    def pipe(self, stream: Iterable[Doc], batch_size: int):
        stream, copy = itertools.tee(stream)
        nes_batch = self.provider.annotate_batch(copy)
        return oies_gen(stream, nes_batch)
