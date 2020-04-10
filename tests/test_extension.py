import spacy
from dynaconf import settings


def test_same_for_equals():
    nlp = spacy.load("en_core_web_sm", disable='ner')
    doc = nlp("I like cakes.")
    assert doc[0:2]._.same(doc[0:2])


def test_not_same_for_different_docs():
    nlp = spacy.load("en_core_web_sm", disable='ner')
    doc_1 = nlp("Cheesecake is great.")
    doc_2 = nlp("Cheesecake is great.")
    assert not doc_1[0:2]._.same(doc_2[0:2])


def test_sanity_check_dynaconf():
    assert settings['test'] == "test123"
