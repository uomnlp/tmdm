from allennlp.data import Instance
from allennlp.models import load_archive
from allennlp.predictors import CorefPredictor, Predictor
from overrides import overrides
from spacy.tokens import Doc
from typing import Iterable, Callable, Any, Dict

from tmdm.classes import OffsetAnnotation, Provider


class OnlinePredictor(Provider):
    def __init__(
            self, task: str, path=None,
            getter: Callable[[Dict[str, Any]], Any] = None,
            preprocessor: Callable[[Doc], Instance] = None,
            converter: Callable[[Doc, Any], OffsetAnnotation] = None
    ):
        super().__init__()

        self.preprocess = preprocessor or self._preprocess
        self.converter = converter or self._converter
        self.getter = getter or None
        self.task = task
        self.path = path
        if self.path:
            self.load(path)

    predictor: Predictor

    @property
    def name(self) -> str:
        return f'allennlp-{self.task}-provider'

    def save(self, path: str):
        ...

    def load(self, path: str):
        self.predictor = Predictor.from_archive(load_archive(path), self.task)

    def annotate_document(self, doc: Doc) -> OffsetAnnotation:
        return self.annotate_batch[0]

    def _preprocess(self, doc: Doc):
        self.predictor._dataset_reader.text_to_instance([[word.text for word in sentence] for sentence in doc.sents])

    def _converter(self):
        ...

    @overrides
    def annotate_batch(self, docs: Iterable[Doc]):
        instances = [self.preprocess(doc) for doc in docs]
        try:
            result = self.predictor.predict_batch_instance(instances)
        except:
            return [[] for _ in docs]
        result_iterator = iter(result)
        return [
            self.converter(self.getter(next(result_iterator)) if self.getter else next(result_iterator)) for _ in docs
        ]
