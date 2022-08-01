from allennlp.data import Instance
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from loguru import logger
from overrides import overrides
from spacy.tokens import Doc
from typing import Callable, Any, Dict, Optional, List

from transformers import Pipeline, pipeline

from tmdm.classes import CharOffsetAnnotation, Provider

default = object()


class OnlineProvider(Provider):
    pipeline: Pipeline

    def __init__(
            self,
            task: str,
            postprocess: bool,
            path_or_name=None,
            path_or_name_tokenizer=None,
            getter: Callable[[Dict[str, Any]], Any] = None,
            preprocessor: Optional[Callable[[Doc], Instance]] = default,
            converter: Callable[[Doc, Any], CharOffsetAnnotation] = None,
            cuda=-1,
    ):
        super().__init__()

        self.postprocess = postprocess
        self.preprocess = self._preprocess if preprocessor is default else preprocessor
        self.converter = converter
        self.getter = getter
        self.task = task
        self.path_or_name = path_or_name
        self.path_or_name_tokenizer = path_or_name_tokenizer or self.path_or_name
        self.cuda = cuda
        self.load()

    @property
    def name(self) -> str:
        return f'transformers-{self.task}-provider'

    def save(self, path: str):
        ...

    def load(self, _=None):
        self.pipeline = pipeline(self.task, model=self.path_or_name, tokenizer=self.path_or_name_tokenizer, device=self.cuda)

    def annotate_document(self, doc: Doc) -> CharOffsetAnnotation:
        return self.annotate_batch([doc])[0]

    def _preprocess(self, doc: Doc):
        return [str(sent) for sent in doc.sents]

    @overrides
    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        instances = [self.preprocess(doc) for doc in docs] if self.preprocess else docs
        # try:
        flat_instances = [i for l in instances for i in l]
        result = self.pipeline(flat_instances)
        logger.trace(f"Result: {result}")
        # except Exception as e:
        #    logger.error(str(e))
        #    return [([], []) for _ in docs]
        result_iterator = iter(result)
        batched_results = [[next(result_iterator) for _ in d.sents] for d in docs]
        batched_results_iter = iter(batched_results)

        return [
            self.converter(doc, self.getter(next(batched_results_iter)) if self.getter else next(batched_results_iter))
            for doc in docs
        ]
