from spacy.gold import offsets_from_biluo_tags
from typing import List

from spacy.tokens import Span

from tmdm.main import tmdm_pipeline
from tmdm.model.ne import NamedEntity

nlp = tmdm_pipeline(getter=None)


def test_same_where_not_equals():
    doc = nlp("Cheesecake is great.")
    doc._.id = "Cheesecake"
    doc._.corefs = [(0, len("Cheesecake"), "CAKE")]
    assert not doc._.corefs[0] == doc[0:1]
    assert doc._.corefs[0]._.same(doc[0:1])


def test_ner_data_model():
    """
    Tests the NER data model...
    """

    tags = ["O", "O", "O", "B-GPE", "I-GPE", "L-GPE", "B-PERSON", "L-PERSON", "O", "O", "O", "U-DATE", "O", "U-GPE",
            "O"]
    doc = nlp("The president of the United States Donald Trump gave a speech today in London.")
    doc._.id = "President"
    doc._.corefs = offsets_from_biluo_tags(doc, tags)
    for ne in doc._.corefs:
        assert isinstance(ne, NamedEntity)
        assert isinstance(ne, Span)

    nes: List[NamedEntity] = doc._.corefs

    assert len(nes) == 4
    assert nes[0].text == "the United States"
    assert nes[0].label_ == "GPE"
    assert nes[1].text == "Donald Trump"
    assert nes[1]._.same(doc[6:8])
