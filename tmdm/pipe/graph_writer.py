import itertools
from uuid import uuid4

from loguru import logger
from spacy.tokenizer import Tokenizer
from typing import Any, Tuple, Callable, Iterable

from spacy.tokens import Doc


class WriterPipe:
    name = "writer-pipe"

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, doc: Doc):
        annotations = self.writer([doc])
        return doc

    def pipe(self, stream: Iterable[Doc], batch_size: int) -> Iterable[Doc]:
        logger.trace(f"Entering pipe, batch_size={batch_size}")
        next_chunk = itertools.islice(stream, batch_size)
        next_chunk = list(next_chunk)
        while next_chunk:
            self.writer(next_chunk)
            next_chunk = itertools.islice(stream, batch_size)
            next_chunk = list(next_chunk)
            for doc in next_chunk:
                yield doc
            next_chunk = itertools.islice(stream, batch_size)
            next_chunk = list(next_chunk)
            logger.debug(next_chunk)
