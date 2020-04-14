from operator import itemgetter

from tmdm.allennlp.common import OnlineProvider
from tmdm.util import convert_clusters_to_offsets


def get_coref_provider(model_path: str):
    from allennlp_models.coref.coref_model import CoreferenceResolver
    converter = convert_clusters_to_offsets
    getter = itemgetter("clusters")
    return OnlineProvider(task='coreference-resolution', path=model_path, converter=converter, getter=getter)
