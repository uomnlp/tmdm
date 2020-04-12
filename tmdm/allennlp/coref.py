from operator import attrgetter

from spacy.tokens import Doc
from typing import List, Tuple

from tmdm.allennlp.common import OnlinePredictor
from tmdm.classes import OffsetAnnotation


def _convert_clusters(doc: Doc, clusters: List[List[Tuple[int, int]]]) -> OffsetAnnotation:
    result = []
    for i, cluster in enumerate(clusters):
        for first_token, last_token in cluster:
            start = doc[first_token].i
            end = doc[last_token].i + len(doc[last_token])
            result.append((start, end, f"CLUSTER-{i}"))
    return result


def get_coref_predictor(model_path: str):
    converter = _convert_clusters
    getter = attrgetter("clusters")
    return OnlinePredictor(task='coref', path=model_path, converter=converter, getter=getter)
