from tmdm import tmdm_scientific_pipeline, add_oie
from tmdm.allennlp.oie import get_oie_provider
from tmdm.util import load_file

docs = load_file("tests/resources/docs.json")
nlp = tmdm_scientific_pipeline()
add_oie(nlp, get_oie_provider("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"))


def test_oie_pipeline_works_with_empty_docs():
    empty_doc = {"id": "empty", "abstract": ""}
    doc = nlp(empty_doc)
    assert doc._.oies == []
