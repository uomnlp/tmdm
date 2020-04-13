from operator import attrgetter, itemgetter

from loguru import logger
from spacy.tokens import Doc
from typing import List, Tuple

from tmdm.allennlp.common import OnlineProvider
from tmdm.classes import OffsetAnnotation


def _convert_clusters(doc: Doc, clusters: List[List[Tuple[int, int]]]) -> OffsetAnnotation:
    result = []
    for i, cluster in enumerate(clusters):
        logger.trace(cluster)
        logger.trace(doc)
        for first_token, last_token in cluster:
            last_token = last_token
            logger.debug(f"doc[{first_token}:{last_token + 1}] = {doc[first_token:last_token + 1]}")
            start = doc[first_token].idx
            end = doc[last_token].idx + len(doc[last_token])
            logger.debug(f"doc.char_span[{start}:{end}] = {doc._.char_span_relaxed(start, end)}")
            result.append((start, end, f"CLUSTER-{i}"))
    return result


def get_coref_provider(model_path: str):
    from allennlp_models.coref.coref_model import CoreferenceResolver
    converter = _convert_clusters
    getter = itemgetter("clusters")
    return OnlineProvider(task='coreference-resolution', path=model_path, converter=converter, getter=getter)
