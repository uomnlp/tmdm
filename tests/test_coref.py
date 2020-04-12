from typing import List
from spacy.tokens import Doc

from tmdm.main import tmdm_pipeline
from tmdm.model.coref import Coreference

nlp = tmdm_pipeline(getter=None)


def test_coref_works():
    txt = "Cheesecake is great. It tastes so good! Another great cake is red velvet cake. I like it too."
    doc: Doc = nlp(txt)
    doc._.id = "Cheesecake"
    # noinspection PyTypeChecker
    doc._.corefs: List[Coreference] = [(0, len("Cheesecake"), "CLUSTER-0"), (txt.index("It"), len("It"), "CLUSTER-0"),
                                       (txt.index("It", 2), len("It"), "CLUSTER-1"),
                                       (txt.index("red velvet cake"), len("red velvet cake"), "CLUSTER-1")]
    assert len(doc._.corefs) == 4
    assert all(c.cluster_id == 0 for c in doc._.corefs[:2])
    assert all(c.cluster_id == 1 for c in doc._.corefs[2:])
    assert doc._.corefs[0] in doc._.corefs[1].cluster
    assert not doc._.corefs[0] in doc._.corefs[3].cluster


def test_not_coreferent_from_different_docs():
    txt = "Cheesecake is great. It tastes so good! Another great cake is red velvet cake. I like it too."
    annotations = [(0, len("Cheesecake"), "CLUSTER-0"), (txt.index("It"), len("It"), "CLUSTER-0"),
                   (txt.index("It", 2), len("It"), "CLUSTER-1"),
                   (txt.index("red velvet cake"), len("red velvet cake"), "CLUSTER-1")]
    doc1 = nlp(txt)
    doc2 = nlp(txt)
    doc1._.corefs = annotations
    doc2._.corefs = annotations
    assert not doc1._.corefs[0]._.same(doc2._.corefs[0])
