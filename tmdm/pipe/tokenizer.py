from uuid import uuid4

from spacy.tokenizer import Tokenizer
from typing import Any, Tuple, Callable

from spacy.tokens import Doc


class IDTokenizer:
    getter: Callable[[Any], Tuple[str, str]]

    def __init__(self, tokenizer: Tokenizer, getter: Callable[[Any, ], Tuple[str, str]]):
        self.tokenizer = tokenizer
        self.getter = getter

    def __call__(self, data: Any, *args, **kwargs) -> Doc:
        if self.getter:
            uuid, text = self.getter(data)
        else:
            uuid = uuid4()
            text = data
        doc: Doc = self.tokenizer.__call__(text, *args, **kwargs)
        doc._.id = str(uuid)
        return doc
