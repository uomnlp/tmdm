# TODO test pipelines
from tmdm.classes import Cached
from tmdm.main import tmdm_pipeline, add_oie
from tmdm.pipe.oie import OIEPipe

testdata = [
    {"id": "doc1", "text": "The cake is a lie. I like trains."},
    {"id": "doc2", "text": "I chew bubble gum and kick butts."}
]


def test_oie_pipeline():
    nlp = tmdm_pipeline(getter=lambda d: (d['id'], d['text']), disable=['ner'])
    provider = Cached(getter=lambda d: d['text'], path='tests/resources/test_oie_pipeline.json')
    add_oie(nlp, provider)
    for doc in nlp.pipe(testdata):
        assert doc._.oies
