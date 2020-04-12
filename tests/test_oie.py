from spacy.gold import offsets_from_biluo_tags
from typing import List

from spacy.tokens import Span

from tmdm.classes import ERTuple
from tmdm.main import tmdm_pipeline, add_ner
from tmdm.model.oie import Verb, Argument

nlp = tmdm_pipeline(getter=None)
txt = "The cake is a lie. I like trains."
annotations = (
    [(9, 11, "VERB"), (0, 8, "ARG"), (12, 17, "ARG"), (21, 25, "VERB"), (19, 20, "ARG"), (26, 32, "ARG")],
    [(0, 1, "ARG-0"), (0, 2, "ARG-1"), (3, 4, "ARG-0"), (3, 5, "ARG-1")]
)
txt2 = "I chew bubble gum and kick butts."
annotations_2 = (
    [(2, 6, "VERB"), (0, 1, "ARG"), (7, 17, "ARG"), (22, 26, "VERB"), (27, 32, "ARG")],
    [(0, 1, "ARG-0"), (0, 2, "ARG-1"), (3, 1, "ARG-0"), (3, 4, "ARG-1")]
)


def test_oie_set_correctly():
    # annotations = [
    #     [(9, 11, "VERB"), (0, 8, "ARG0"), (13, 17, "ARG1")],
    #     [(21, 25, "VERB"), (19, 20, "ARG0"), (26, 32, "ARG1")]
    # ]

    doc = nlp(txt)
    doc._.oies = annotations
    assert doc._._oies == annotations


def test_oie_get_correctly():
    doc = nlp(txt)
    doc._.oies = annotations
    assert doc._._oies == annotations
    verbs = doc._.oies
    assert len(verbs) == 2
    assert verbs[0]._.same(doc[2:3]), f"{verbs[0]} != {doc[2:3]}"
    assert verbs[1]._.same(doc[7:8]), f"{verbs[1]} != {doc[7:8]}"


def test_oie_gets_arguments_correctly():
    doc = nlp(txt)
    doc._.oies = annotations
    verbs: List[Verb] = doc._.oies
    assert len(verbs) == 2
    verb0args: List[Argument] = verbs[0].arguments
    assert len(verb0args) == 2
    arg0: Argument = verb0args[0]
    assert arg0._.same(doc[0:2])
    arg1: Argument = verb0args[1]
    assert arg1._.same(doc[3:5])


def test_arg_mapping_works_backwards():
    doc = nlp(txt)
    doc._.oies = annotations
    arg0 = doc[0]._.oies[0]
    assert isinstance(arg0, Argument)
    assert isinstance(arg0.verbs[0], Verb)
    assert arg0.verbs[0]._.same(doc[2:3])


def test_arg_mapping_works_with_multiple_verbs():
    doc = nlp(txt2)
    doc._.oies = annotations_2
    verbs: List[Verb] = doc._.oies
    assert len(verbs) == 2
    verb0args: List[Argument] = verbs[0].arguments
    arg0: Argument = verb0args[0]
    verb1args: List[Argument] = verbs[1].arguments
    arg1 = verb1args[0]
    assert arg0 == arg1


def test_continuous():
    doc = nlp(txt2)
    doc._.oies = annotations_2
    verbs: List[Verb] = doc._.oies
    assert verbs[0].continuous
    assert not verbs[1].continuous
    assert verbs[0].extended_span.text == "I chew bubble gum"
