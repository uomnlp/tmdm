from operator import itemgetter

from tmdm.allennlp.common import OnlineProvider
from tmdm.pipe.pipe import PipeElement
from tmdm.util import convert_clusters_to_offsets


def get_coref_provider(model_path: str):
    from allennlp_models.coref import CoreferenceResolver
    converter = convert_clusters_to_offsets
    getter = itemgetter("clusters")
    return OnlineProvider(task='coreference-resolution', path=model_path, converter=converter, getter=getter)


def get_coref_pipe(model_path: str = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'):
    from allennlp_models.pretrained import CoreferenceResolver
    converter = convert_clusters_to_offsets
    getter = itemgetter("clusters")
    return PipeElement(name='coref', field='corefs',
                       provider=OnlineProvider(task='coreference_resolution', path=model_path, converter=converter, getter=getter))
