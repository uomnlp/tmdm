import spacy
from dynaconf import settings

from tmdm.classes import ERTuple
from tmdm.main import tmdm_pipeline

txt = "I like cakes. They taste nice."
nlp = tmdm_pipeline()
doc = nlp(txt)
doc._.corefs = [(txt.index('cakes'), txt.index('cakes') + len('cakes'), "CLUSTER-0"),
                (txt.index('They'), txt.index('They') + len("They"), "CLUSTER-0")]
doc._.oies = ERTuple([(txt.index('I'), txt.index('I') + len('I'), "ARG-0"),
                      (txt.index('like'), txt.index('like') + len("like"), "VERB"),
                      (txt.index('cakes'), txt.index('cakes') + len('cakes'), "ARG-1")],
                     [(1, 0, "ARG-0"), (1, 2, "ARG-1")])
doc._.nes = [(txt.index('cakes'), txt.index('cakes') + len('cakes'), "FOOD")]


def test_get_coref_works():
    # sanity check
    assert len(doc._.corefs) == 2
    assert doc[2:3]._.is_coref()
    assert doc[2:3]._.get_coref().coreferent(doc[4:5]._.get_coref())


def test_get_oie_works():
    # sanity check
    assert len(doc._._oies.entities) == 3
    assert doc[1:2]._.is_verb()
    assert doc[0:1]._.is_argument()
    assert doc[2:3]._.get_argument().text == "cakes"


def test_get_ne_works():
    # sanity check
    assert len(doc._.nes) == 1
    assert doc[2:3]._.is_ne()
    assert doc[2:3]._.get_ne().kb_id_ == "FOOD/cakes"


def test_coref_works_with_oie():
    assert doc._.corefs[0]._.is_argument()
    assert doc._.oies[0].arguments[1]._.get_coref().coreferent(doc[4:5]._.get_coref())


def test_ne_works_with_coref():
    assert doc[2:3]._.is_ne()
    assert doc[2:3]._.get_ne()._.get_coref().coreferent(doc[4:5]._.get_coref())
