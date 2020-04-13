from typing import Dict, Any, Union, Callable

from loguru import logger
from overrides import overrides
from spacy.gold import offsets_from_biluo_tags, iob_to_biluo
from spacy.tokens.doc import Doc

from tmdm import util
from tmdm.classes import Provider, OffsetAnnotation
from tmdm.util import convert_clusters_to_offsets, get_offsets, get_offsets_from_sentences

OFFSETS = object()


class Cached(Provider):
    cache: Dict[str, Any]
    name = 'cached'
    known_schemas = {
        # these assume same tokenisation
        "bio": lambda doc, annotation: offsets_from_biluo_tags(iob_to_biluo(doc, annotation)),
        "bilou": offsets_from_biluo_tags,
        "offsets": OFFSETS,
        "list_of_clusters": convert_clusters_to_offsets,
        # these provide their own tokenisation

        # annotation: List[Tuple[str,str]]
        "list_of_tuples_bio_flat": lambda doc, annotation: get_offsets(doc.text, annotation),

        # annotation: List[List[Tuple[str,str]]]
        "list_of_tuples_bio_stacked": lambda doc, annotation: get_offsets_from_sentences(doc.text, annotation),

        # annotation: Tuple[List[str],List[str]]
        "tuple_of_lists_flat": lambda doc, annotation: get_offsets(doc.text, zip(*annotation[:2])),

        # annotation: List[Tuple[List[str]], Tuple[List[str]]]
        "list_of_tuples_of_lists": lambda doc, annotation:
        get_offsets_from_sentences(doc.text, ((w, l) for t in annotation for w, l in zip(*t[:2]))),

        # annotation: Tuple[List[List[str]], Tuple[List[List[str]]
        "tuple_of_lists_of_lists": lambda doc, annotation:
        get_offsets_from_sentences(doc.text, ((w, l) for ws, ls in zip(*annotation[:2]) for w, l in zip(ws, ls)))

        # TODO: BRAT
        # TODO: Pubmed
    }

    def __init__(self, schema: Union[str, Callable[[Doc, Any], OffsetAnnotation]] = None, getter=None,
                 path: str = None):
        self.cache = {}
        self.loaded = False
        if not schema:
            self.schema = OFFSETS
        elif schema in self.known_schemas:
            self.schema = Cached.known_schemas[schema]
        elif isinstance(schema, Callable):
            self.schema = schema
        else:
            self.schema = None
        self.getter = getter
        if path:
            self.load(path)

    @overrides
    def save(self, path: str):
        util.save_file(self.cache, path)

    # TODO: guess schema

    @overrides
    def load(self, path):
        self.cache = util.load_file(path)
        self.loaded = True

    @overrides
    def annotate_document(self, doc: Doc) -> OffsetAnnotation:
        if not self.loaded:
            raise ValueError("You forgot to load the cache!")
        annotations = self.cache.get(doc._.id, None)
        if annotations:
            if self.schema:
                if self.schema == OFFSETS:
                    return self.getter(annotations) if self.getter else annotations
                else:
                    return self.schema(doc, self.getter(annotations) if self.getter else annotations)
            else:
                logger.info(f"no schema loaded for {self.__class__.__name__}, good luck!")
                return annotations
