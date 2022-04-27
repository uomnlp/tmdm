from uuid import uuid4

from spacy.tokenizer import Tokenizer
from typing import Any, Tuple, Callable

from spacy.tokens import Doc


class IDAssigner:
    getter: Callable[[Any], Tuple[str, str]]

    def __init__(self, tokenizer: Tokenizer, getter: Callable[[Any, ], Tuple[str, str]], generate_ids=True):
        self.tokenizer = tokenizer
        self.getter = getter
        self.generate_ids =generate_ids

    def __call__(self, doc: Doc, *args, **kwargs) -> Doc:
        if not self.generate_ids:
            uuid, text = self.getter(doc)
        else:
            uuid = uuid4()
            text = self.getter(doc) if self.getter else doc
        #doc: Doc = self.tokenizer.__call__(text, *args, **kwargs)
        doc._.id = str(uuid)
        return doc
