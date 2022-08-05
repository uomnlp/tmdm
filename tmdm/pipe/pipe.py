import itertools
from typing import Iterable, Any, List, Callable, Union, Collection
from uuid import uuid4

from loguru import logger
from spacy.tokens import Doc

from tmdm.classes import Provider


class Pipeline:
    def __init__(self, nlp, pipes=None, getter=None, generate_ids=True):
        self.pipes: List[PipeElement] = pipes or []
        self.nlp = nlp
        self.getter = getter
        self.generate_ids = generate_ids
        if not self.generate_ids and not getter:
            raise ValueError("Need IDs from somewhere!")

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def remove_pipe(self, name: str):
        to_remove = next(p for p in self.pipes if p.name == name)
        if not to_remove:
            raise ValueError(f"Pipe {name} not in pipeline!")
        else:
            self.pipes.remove(to_remove)
            return to_remove

    def preprocess(self, data: Any) -> Doc:
        if not self.generate_ids:
            uuid, text = self.getter(data)
        else:
            uuid = uuid4()
            text = self.getter(data) if self.getter else data
        doc = self.nlp(text)
        doc._.id = uuid
        return doc

    def __call__(self, data: Any) -> Doc:
        doc = self.preprocess(data)
        for p in self.pipes:
            p(doc)
        return doc

    def pipe(self, data_stream: Iterable[Any], batch_size=8) -> Iterable[Doc]:
        data_stream = iter(data_stream)
        chunk = itertools.islice(data_stream, batch_size)
        chunk = list(chunk)
        while chunk:
            chunk = list([self.preprocess(d) for d in chunk])
            for pipe in self.pipes:
                pipe.pipe(chunk)
            for doc in chunk:
                yield doc
            chunk = itertools.islice(data_stream, batch_size)
            chunk = list(chunk)


class PipeElement:
    def __init__(self, name, field, provider: Union[Provider, Callable[[Collection[Doc], ], Any]]):
        self.name = name
        self.provider = provider
        self.field = field

    def __call__(self, doc: Doc):
        try:
            annotations = self.provider.annotate_document(doc)
        except AttributeError:
            annotations = self.provider([doc])
        logger.debug(annotations)
        if self.field:
            setattr(doc._, self.field, annotations)

    def pipe(self, docs: List[Doc]):
        try:
            annotated_batch = self.provider.annotate_batch(docs)
        except AttributeError:
            annotated_batch = self.provider(docs)
        logger.debug(annotated_batch)
        if self.field:
            assert len(docs) == len(annotated_batch)
            for doc, annotations in zip(docs, annotated_batch):
                setattr(doc._, self.field, annotations)

            if self.name == 'ner' and self.provider.post_process:
                allnestuples = self.provider.postprocess_batch(docs)

                for doc in docs:
                    for ne in doc._.nes:
                        ne.cache.clear()

                for doc, nestuples in zip(docs, allnestuples):
                    setattr(doc._, self.field, nestuples)
