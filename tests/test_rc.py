from spacy.tokens import Doc
from tmdm.main import tmdm_pipeline
from tmdm.model.rc import Relation


nlp = tmdm_pipeline()
txt = "John has been in a marriage with Jane since 1983."
annotations = [(0,4,"0-subj-per:spouse"),(33,37,"0-obj-per:spouse")]


def test_rc_set_correctly():
    doc = nlp(txt)
    doc._.relations = annotations
    assert doc._._relations == annotations
    relation = doc._.relations[0]
    assert isinstance(relation, Relation)


def test_rc_get_attributes():
    doc = nlp(txt)
    doc._.relations = annotations
    relation = doc._.relations[0]
    assert relation.idx == 0
    assert relation.relation_type == "per:spouse"
    assert relation.subject._.same(doc[0:1])
    assert relation.object._.same(doc[7:8])