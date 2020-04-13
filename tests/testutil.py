import ujson as json

from tmdm.allennlp.coref import get_coref_provider
from tmdm.allennlp.oie import get_oie_provider
from tmdm.main import tmdm_pipeline, add_ner, add_coref, add_oie


def get_test_docs():
    with open('tests/resources/docs.json', encoding='utf-8') as f:
        return json.load(f)


def get_test_ner():
    with open('tests/resources/ner.json', encoding='utf-8') as f:
        return json.load(f)


def get_coref_oie_online_pipeline():
    nlp = tmdm_pipeline()
    add_coref(nlp,
              get_coref_provider(
                  "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
              )
    add_oie(nlp,
            get_oie_provider("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
            )
    return nlp
