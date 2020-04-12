# TODO test pipelines
from tmdm.classes import Cached
from tmdm.main import tmdm_pipeline, add_oie, add_coref

testdata = [
    {"id": "doc1", "text": "The cake is a lie. I like trains."},
    {"id": "doc2", "text": "I chew bubble gum and kick butts."}
]

testdata2 = [
    {"Cheesecake is great. It tastes so good!"},
    {"id": "doc1", "text": "The cake is a lie. I like trains."}
]

def test_oie_pipeline():
    nlp = tmdm_pipeline(getter=lambda d: (d['id'], d['text']), disable=['ner'])
    provider = Cached(getter=lambda d: d['text'], path='tests/resources/test_oie_pipeline.json')
    add_oie(nlp, provider)
    for doc in nlp.pipe(testdata):
        assert doc._.oies


def test_coref_pipeline():
    nlp = tmdm_pipeline(getter=lambda d: (d['id'], d['text']), disable=['ner'])
    provider = Cached(getter=lambda d: d['text'], path='tests/resources/test_oie_pipeline.json')
    add_coref(nlp, provider)
    for doc in nlp.pipe(testdata):
        assert doc._.corefs