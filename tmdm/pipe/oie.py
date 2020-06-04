import itertools
from loguru import logger
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

    def grouper(self, n, iterable):
        it = iterable
        while True:
            # logger.trace(f"{len(list(iterable))} docs in stream {type(iterable)}, batch_size: {n}")
            chunk = list(itertools.islice(it, n))
            logger.trace(f"Chunk: {chunk}")
            if not chunk:
                return
            yield self.provider.annotate_batch(chunk)

    def pipe(self, stream: Iterable[Doc], batch_size: int) -> Iterable[Doc]:
        logger.trace(f"Entering pipe, batch_size={batch_size}")
        next_chunk = itertools.islice(stream, batch_size)
        next_chunk = list(next_chunk)
        while next_chunk:
            annotated_batch = self.provider.annotate_batch(next_chunk)
            assert len(next_chunk) == len(annotated_batch)
            for doc, annotations in zip(next_chunk, annotated_batch):
                doc._.oies = annotations
                yield doc
            next_chunk = itertools.islice(stream, batch_size)
            next_chunk = list(next_chunk)
            logger.debug(next_chunk)
