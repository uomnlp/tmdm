import itertools
from spacy.tokens import Doc
from typing import Iterable

from tmdm.classes import Provider


def doc_gen(stream, nes_batch):
    for doc, corefs in zip(stream, nes_batch):
        doc._.corefs = corefs
        yield doc


class CorefPipe:
    name = "coref-pipe"

    def __init__(self, vocab, provider: Provider):
        self.vocab = vocab
        self.provider = provider

    def __call__(self, doc: Doc):
        annotations = self.provider.annotate_document(doc)
        doc._.corefs = annotations
        return doc

    def pipe(self, stream: Iterable[Doc], batch_size: int):
        stream, copy = itertools.tee(stream)
        coref_batch = self.provider.annotate_batch(copy)
        return doc_gen(stream, coref_batch)
