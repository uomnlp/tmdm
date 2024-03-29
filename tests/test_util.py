import string

from loguru import logger

from tests import testutil
from tmdm.util import merge_two_annotations, bio_generator, get_offsets, get_offsets_from_sentences, get_offsets_from_brat
import pytest
import os

test_annotation_none = "O O O O O".split()
test_annotation_begin = "B-A O O O O".split()
test_annotation_begin_2 = "B-A I-A O O O".split()
test_annotation_middle = "O B-A I-A O O".split()
test_annotation_middle_two = "O O B-X I-X O".split()
test_annotation_end = "O O O B-X I-X".split()
test_annotation_wrong_length = "O B-A I-A".split()
test_annotation_end_single = "O O O O B-A".split()
test_annotation_two_annotations = "B-A I-A O O B-A".split()


def test_bio_generator_empty():
    assert list(bio_generator(test_annotation_none)) == []


def test_bio_generator_single_tag():
    # single tag
    assert list(bio_generator(test_annotation_begin)) == [((0, 1), "A")]
    assert list(bio_generator(test_annotation_end)) == [((3, 5), "X")]
    assert list(bio_generator(test_annotation_end_single)) == [((4, 5), "A")]


def test_merge_two_bio_annotations_sanity_check():
    with pytest.raises(AssertionError):
        merge_two_annotations([test_annotation_middle_two], [test_annotation_wrong_length])
    with pytest.raises(AssertionError):
        merge_two_annotations([test_annotation_wrong_length], [test_annotation_middle_two])
    with pytest.raises(AssertionError):
        merge_two_annotations([test_annotation_middle_two, test_annotation_begin], [test_annotation_middle_two])


def test_merge_two_different():
    result = merge_two_annotations([test_annotation_middle_two], [test_annotation_none])
    assert result == [test_annotation_middle_two]
    result = merge_two_annotations([test_annotation_begin], [test_annotation_end])
    assert result == ["B-A O O B-X I-X".split()]


def test_merge_two_overlapping():
    result = merge_two_annotations([test_annotation_middle_two], [test_annotation_middle])
    assert result == [test_annotation_middle_two]
    result = merge_two_annotations([test_annotation_begin_2], [test_annotation_two_annotations])
    assert result == ["B-A I-A O O B-A".split()]


def test_get_offsets_works_with_sane_text():
    text = "I like cakes."
    annotation = list(zip('I like cakes .'.split(), "O O B-CAKE O".split()))
    assert get_offsets(text, annotation) == [(7, 12, "CAKE")]


def test_get_offsets_works_with_wrong_capitalisation():
    text = "I like cakes."
    annotation = list(zip('i like cakes .'.split(), "O O B-CAKE O".split()))
    assert get_offsets(text, annotation) == [(7, 12, "CAKE")]


def test_get_offsets_works_with_bio_tags():
    text = "I like big cakes."
    annotation = list(zip('I like big cakes .'.split(), "O O B-CAKE I-CAKE O".split()))
    assert get_offsets(text, annotation) == [(7, 16, "CAKE")]


def test_get_offsets_works_with_consecutive_tags():
    text = "I like wedding cake cakes."
    annotation = list(zip('I like wedding cake cakes .'.split(), "O O B-CAKE I-CAKE B-CAKE O".split()))
    assert get_offsets(text, annotation) == [(7, 19, "CAKE"), (20, 25, "CAKE")]


def test_get_offsets_works_with_last_tag():
    logger.info(f"Working dir: {os.getcwd()}")
    text = "I like cakes"
    annotation = list(zip(text.split(), "O O B-CAKE".split()))
    assert get_offsets(text, annotation) == [(7, 12, "CAKE")]


def test_get_offsets_works_with_last_longer_tags():
    text = "I like big cakes"
    annotation = list(zip(text.split(), "O O B-CAKE I-CAKE".split()))
    assert get_offsets(text, annotation) == [(7, 16, "CAKE")]


def test_get_offsets_works_with_commas_in_between():
    text = "I like, wedding cake cakes."
    annotation = list(zip('I like , wedding cake cakes .'.split(), "O O O B-CAKE I-CAKE B-CAKE O".split()))
    assert get_offsets(text, annotation) == [(8, 20, "CAKE"), (21, 26, "CAKE")]


def test_get_offsets_works_with_funky_spacing():
    text = "I like ,    wedding cake cake."
    annotation = list(zip('I like , wedding cake cake .'.split(), "O O O B-CAKE I-CAKE B-CAKE O".split()))
    assert get_offsets(text, annotation) == [(12, 24, "CAKE"), (25, 29, "CAKE")]


def test_offset_latch_match_returns_position_of_last_token():
    text = "I like ,    wedding cake cake troll"
    annotation = list(zip('I like , wedding cake cake troll'.split(), "O O O B-CAKE I-CAKE B-CAKE O".split()))
    _, last_match = get_offsets(text, annotation, return_last_match=True)
    assert last_match == 35


def test_get_sent_offsets_works_with_sane_input():
    text = "I like cakes. Cheese cake is my favourite."
    annotation = [list(zip("I like cakes .".split(), "O O B-CAKE O".split())),
                  list(zip("Cheese cake is my favourite .".split(), "B-CAKE I-CAKE O O O O".split()))]
    offsets = get_offsets_from_sentences(text, annotation)
    assert text[slice(*offsets[0][:2])] == 'cakes' and text[slice(*offsets[1][:2])] == 'Cheese cake'


def test_get_sent_offsets_works_with_funky_spacing_input():
    text = "I        like\tcakes   .  Cheese cake     is  my  favourite cake."
    annotation = [list(zip("I like cakes .".split(), "O O B-CAKE O".split())),
                  list(zip("Cheese cake is my favourite cake .".split(), "B-CAKE I-CAKE O O O B-CAKE O".split()))]
    offsets = get_offsets_from_sentences(text, annotation)
    assert text[slice(*offsets[0][:2])] == 'cakes'
    assert text[slice(*offsets[1][:2])] == 'Cheese cake'
    assert text[slice(*offsets[2][:2])] == 'cake'


def test_get_sent_offsets_on_realistic_input():
    docs = testutil.get_test_docs()
    annotations = testutil.get_test_ner()

    def normalise(input: str):
        chars = (set(string.printable) - set(string.whitespace))
        return "".join(c.lower() for c in input if c in chars)

    for doc in docs:
        for field in ["title", "abstract"]:
            if doc[field]:
                idx = doc['id']
                offsets = get_offsets_from_sentences(doc[field], annotations[idx][field])
                all_anns = []
                for sentence in annotations[idx][field]:
                    tokens = []
                    anns = []
                    for t, ann in sentence:
                        tokens.append(t)
                        anns.append(ann)

                    labels = list(bio_generator(anns))
                    for (s, e), _ in labels:
                        all_anns.append("".join(normalise(t) for t in tokens[s:e]))

                assert len(all_anns) == len(offsets), doc['title']

                for i, (start, end, _) in enumerate(offsets):
                    assert all_anns[i] == normalise(doc[field][start:end])

def test_get_brat_offsets_works_with_links():
    brat_ann = """T1	Location 57 65	Spratton
#1	AnnotatorNotes T1	https://en.wikipedia.org/wiki/Spratton"""
    
    brat_ann = brat_ann.split("\n")
    print(list(get_offsets_from_brat(brat_ann)))
    assert get_offsets_from_brat(brat_ann) == [(57, 65, {"label": "Location", "URI": "https://en.wikipedia.org/wiki/Spratton"})]
               
def test_get_brat_offsets_works_with_sane_input():
    text = "John was born in Spratton in 1890, the son of Joseph Copson"
    annotation = ["T1	Person 46 59	Joseph Copson"]
    offsets, label = get_offsets_from_brat(annotation, True)
    assert text[slice(*offsets[0][:2])] == 'Joseph Copson'

def test_get_brat_offsets_on_realistic_input():
    
    def normalise(input: str):
        chars = (set(string.printable))
        return "".join(c.lower() for c in input if c in chars)
    
    brat_txt, brat_ann = testutil.get_test_brat()
    brat_ann = brat_ann.split("\n")
    
    norm_txt = list(map(normalise, brat_txt))
    offsets, label = get_offsets_from_brat(brat_ann, True)
    
    for idx, s in enumerate(offsets):
        token = norm_txt[s[0]:s[1]]
        assert "".join(token) == label[idx]
