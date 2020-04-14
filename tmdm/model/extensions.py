import itertools
from collections import defaultdict
from typing import Tuple, Union, Callable, Dict, List, Type

from dynaconf import settings
from fastcache import clru_cache
from loguru import logger
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

force = settings.get('force', False)


def extend(target: Union[Doc, Token, Span], type='method', setter=None, default=None, create_attribute=True):
    def inner(func: Callable):
        inner.__name__ = func.__name__
        inner.__qualname__ = func.__qualname__
        inner.__doc__ = func.__doc__
        name = func.__name__
        logger.trace(f"Creating extension for '{target.__name__}' called '{name}' of type '{type}'")

        if type == 'method':
            target.set_extension(name, method=func, force=force)
        if type == 'property':
            if create_attribute:
                logger.trace(f"Creating attribute '_{name}'")
                target.set_extension("_" + name, default=default, force=force)
            target.set_extension(name, getter=func, force=force, setter=setter)
        return func

    return inner


# ATTRIBUTES
Doc.set_extension('id', default=None, force=force)


# PROPERTIES
@extend(Doc, 'property', create_attribute=True)
def token_map(self: Doc):
    # TODO: another candidate for porting to faster code
    if not self._._token_map:
        token_map = []
        for i, token in enumerate(self):
            token_map.extend([i] * (len(token) + (1 if token.whitespace_ else 0)))
        self._._token_map = token_map
    return self._._token_map


# METHODS
@extend(Doc)
def to_relative(self: Doc, start, end) -> Tuple[int, Tuple[int, int]]:
    sent_start = self[start].sent[0].i
    sent_nr = get_sent_nr(self[start].sent)
    return sent_nr, (start - sent_start, end - sent_start)


@extend(Span)
def same(self: Span, other: Span):
    """
    Equality based on the exact match of the spans.

    Args:
        self: will be filled by partial
        other: Other extraction to compare to.

    Returns:
      True if spans match exactly, False otherwise.

    """
    return (self.doc == other.doc  # same doc
            and self.start == other.start
            and self.end == other.end)


@extend(Doc)
@clru_cache(maxsize=100000)
def char_span_relaxed(self: Doc, start: int, end: int):
    logger.trace(f"Computing not cached doc.char_span({start},{end})")
    span = self.char_span(start, end)
    if not span:
        token_map = self._.token_map
        span = self[token_map[start]:token_map[end] + 1]
    return span


@extend(Span)
def isin(self: Span, item: Union[str, Span]):
    """
    Whether one span is subsumed by another one.

    Args:
        self: will be filled by partial
        item: Span or string to be compared.

    Returns: True if this span is subsumed by the given one.

    """
    if isinstance(item, str):
        return self.text in item
    elif isinstance(item, Span):
        return self.doc == item.doc and item.start <= self.start and item.end >= self.end
    else:
        return False


@extend(Doc)
def to_absolute(self: Doc, sent_nr: int, start: int, end: int) -> Tuple[int, int]:
    first_idx = get_sent(sent_nr)[0].i
    return start + first_idx, end + first_idx


@extend(Span)
def contains(self: Span, item: Union[str, Span]):
    if isinstance(item, str):
        return item in self.text
    else:
        return isin(item, self)


@extend(Doc)
def get_sent_nr(self, sent: Span):
    for i, s in enumerate(self.sents):
        if s == sent:
            return i
    raise ValueError(f"'{sent}' is not in '{self}'! ")


@extend(Doc)
def get_sent(self, n: int):
    return next(itertools.islice(self.sents, n, n + 1))


class Annotation(Span):
    idx: int = None
    cache: Dict[Type, Dict[str, Dict[int, List]]] = defaultdict(lambda: defaultdict(dict))

    @property
    def fqn(self):
        return f"{self.doc._.id}/{str(self.__class__.__name__)}/{self.idx} ('{str(self)}')"

    @classmethod
    def make(cls, doc: Doc, idx: int, start, end, label, *args, **kwargs):
        if not doc._.id or not cls.cache[cls][doc._.id].get(idx, None):

            span = doc._.char_span_relaxed(start, end)
            span = cls(doc, span.start, span.end, label, *args, **kwargs)
            span.idx = idx
            logger.trace(f"creating {span.fqn}")
            if doc._.id:
                cls.cache[cls][doc._.id][idx] = span
            else:
                logger.trace('Id not set for doc, omit caching...')
        else:
            logger.trace(f"{cls.cache[cls][doc._.id][idx]} is in cache!")
        return cls.cache[cls][doc._.id][idx]
