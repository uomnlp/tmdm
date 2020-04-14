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


def test_contains_coref_works():
    assert not doc[2:4]._.is_coref()
    assert doc[2:4]._.has_corefs()
    corefs = doc[2:4]._.get_corefs()
    assert len(corefs) == 1
    assert doc[2:4].text == 'cakes.'
    assert corefs[0].text == 'cakes'


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


def test_contains_ne_works():
    assert not doc[2:4]._.is_ne()
    assert doc[2:4]._.has_nes()
    nes = doc[2:4]._.get_nes()
    assert len(nes) == 1
    assert doc[2:4].text == 'cakes.'
    assert nes[0].text == 'cakes'


def test_contains_oie_works():
    # sanity check
    assert len(doc._._oies.entities) == 3
    assert not doc[1:3]._.is_verb()
    assert doc[1:3]._.has_verbs()
    verbs = doc[1:3]._.get_verbs()
    assert len(verbs) == 1
    assert verbs[0].text == 'like'
    assert not doc[:3]._.is_argument()
    assert doc[:3]._.has_arguments()
    arguments = doc[:3]._.get_arguments()
    assert len(arguments) == 2
    assert arguments[0].text == "I"
    assert arguments[1].text == "cakes"
