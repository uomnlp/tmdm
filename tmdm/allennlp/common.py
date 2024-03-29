from allennlp.data import Instance
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from loguru import logger
from overrides import overrides
from spacy.tokens import Doc
from typing import Callable, Any, Dict, Optional, List
from tmdm.classes import CharOffsetAnnotation, Provider

default = object()


class OnlineProvider(Provider):
    def __init__(
            self, task: str, path=None,
            getter: Callable[[Dict[str, Any]], Any] = None,
            preprocessor: Optional[Callable[[Doc], Instance]] = default,
            converter: Callable[[Doc, Any], CharOffsetAnnotation] = None,
            cuda=-1,
    ):
        super().__init__()

        self.preprocess = self._preprocess if preprocessor is default else preprocessor
        self.converter = converter
        self.getter = getter
        self.task = task
        self.path = path
        self.cuda = cuda
        if self.path:
            self.load(path)

    predictor: Predictor

    @property
    def name(self) -> str:
        return f'allennlp-{self.task}-provider'

    def save(self, path: str):
        ...

    def load(self, path: str):
        self.predictor = Predictor.from_archive(load_archive(path, cuda_device=self.cuda), self.task)

    def annotate_document(self, doc: Doc) -> CharOffsetAnnotation:
        return self.annotate_batch([doc])[0]

    def _preprocess(self, doc: Doc):
        return self.predictor._dataset_reader.text_to_instance(
            [[word.text for word in sentence] for sentence in doc.sents]
        )

    @overrides
    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        instances = [self.preprocess(doc) for doc in docs] if self.preprocess else docs
        # try:
        result = self.predictor.predict_batch_instance(instances)
        logger.trace(f"Result: {result}")
        # except Exception as e:
        #    logger.error(str(e))
        #    return [([], []) for _ in docs]
        result_iterator = iter(result)

        return [
            self.converter(doc, self.getter(next(result_iterator)) if self.getter else next(result_iterator))
            for doc in docs
        ]
