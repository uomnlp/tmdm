import string

import handystuff.loaders

import tmdm
# from scispacy.custom_sentence_segmenter import combined_rule_sentence_segmenter
from spacy.tokens import Doc
from srsly import msgpack
from srsly import ujson as json
from srsly import cloudpickle as pickle
from typing import List, Iterable

from loguru import logger

from typing import Tuple

from tmdm.classes import CharOffsetAnnotation

ACCEPTED_FORMATS = ('mpk', 'json', 'pkl', 'jsonl')


def _check_fmt(path, format=None):
    if not format:
        try:
            format = path.rsplit('.')[-1]
        except IndexError:
            raise ValueError(f"Cannot automatically infer file format from file path '{path}'! "
                             f"Provide explicit format from [{ACCEPTED_FORMATS}]")
    if not format in ACCEPTED_FORMATS:
        raise ValueError(f"Only serialisation formats from [{ACCEPTED_FORMATS}] are supported!")
    return format


def save_file(file, path: str, format: str = None):
    format = _check_fmt(path, format)
    if format == 'mpk':
        with open(path, "wb+") as f:
            msgpack.dump(file, f)
    elif format == "json":
        with open(path, 'w+') as f:
            json.dump(file, f)
    elif format == "pkl":
        with open(path, 'wb+') as f:
            pickle.dump(file, f)


def load_file(path: str, format: str = None):
    format = _check_fmt(path, format)
    if format == 'mpk':
        with open(path, "rb") as f:
            return msgpack.load(f)
    elif format == 'jsonl':
        return handystuff.loaders.load_jsonl(path)
    elif format == "json":
        with open(path, 'r') as f:
            return json.load(f)
    elif format == "pkl":
        with open(path, 'rb') as f:
            return pickle.load(f)


def as_printable(text):
    return ''.join(c for c in text if c in string.printable)


def get_offsets(text: str, annotation: Iterable[Tuple[str, str]], init_offset=0, return_last_match=False):
    result = []
    offset = 0
    text = text.lower()

    for token, label in annotation:
        try:
            start = text[offset:].index(token.lower()) + offset
            end = start + len(token)
            logger.trace(f"searching for '{token}' ({label}) in '{text[offset:offset + end + 20]} [...]'")
            logger.trace(f'text[{start}:{end}]: \'{text[start:end]}\'')

        except ValueError:
            # raise NotImplementedError()
            logger.trace(f"{token} not in {text[offset:offset + 20]}!")
            printable_text = as_printable(text[offset:])
            try:
                matched_index = printable_text.index(token.lower())
                start = matched_index + offset
                matched_string = printable_text[matched_index:matched_index + len(token)]
                char_iter = iter(text[start:])
                diff = 0
                logger.trace(f"Matched {matched_string}")
                for t in matched_string:
                    original_char = next(char_iter)
                    while t != original_char:
                        original_char = next(char_iter)
                        diff += 1

                logger.trace(f"diff is {diff}")
                end = start + len(token) + diff
                logger.trace(f"searching for '{token}' ({label}) in '{printable_text[:offset + end + 20]} [...]'")
                logger.trace(f'text[{start}:{end}]: \'{text[start:end]}\'')
            except ValueError:
                logger.trace(f"{token} not in {printable_text[:offset + len(token)]}!")
                raise NotImplementedError

        if not label == "O":
            tag, category = label.split('-')
            if tag == "B":
                result.append((start + init_offset, end + init_offset, category))
            elif tag == "I":
                old_start, *rest = result[-1]
                result[-1] = (old_start, end + init_offset, category)
        offset = end
    if return_last_match:
        return result, offset + init_offset
    else:
        return result


def get_offsets_from_sentences(text: str, annotation: Iterable[Iterable[Tuple[str, str]]]):
    offset = 0
    result = []
    for sent in annotation:
        logger.trace(text[offset:])
        sent_result, offset = get_offsets(text[offset:], sent, init_offset=offset, return_last_match=True)
        result.extend(sent_result)
    return result


def get_offsets_from_brat(annotation: Iterable[str], testing: bool = False):
    tags = [a.split("\t") for a in annotation if a.strip()[0] == "T"]
    links = [a.split("\t") for a in annotation if a.strip()[0] == "#"]
    result = {}
    for t in tags:
        offset = t[1].split()
        result[t[0]] = [int(offset[1]), int(offset[2]), {"label": offset[0], "URI": ""}]
    for l in links:
        idx = l[1].split()[1]
        result[idx][2]["URI"] = l[2]
    ret = [tuple(x) for x in result.values()]
    if testing:
        label = [a[2].lower() for a in tags]
        return [ret, label]
    return ret


def bio_generator(tags: List[str], sep='-') -> Tuple[Tuple[int, int], str]:
    start = 0
    for i, tag in enumerate(tags):
        eos_or_eot = i + 1 >= len(tags) or not tags[i + 1].startswith("I")

        if tag != "O":
            chunk, category = tag.split(sep)
            if chunk == "B":
                if eos_or_eot:
                    yield (i, i + 1), category
                else:
                    start = i
            elif chunk == "I":
                # if end of sentence OR end of the tag
                if eos_or_eot:
                    # U/L always preceded by B.
                    yield (start, i + 1), category


def entities_relations_from_by_verb(
        by_verb: List[List[Tuple[int, int, str]]]) -> 'tmdm.classes.EntitiesRelationshipsAnnotation':
    entities = []
    relations = []
    for verb_annotations in by_verb:
        verb = next((start, end, label) for start, end, label in verb_annotations if label == 'V' or label == 'VERB')
        arg_annotations = ((s, e, l) for s, e, l in verb_annotations if not (l == 'V' or l == 'VERB'))
        entities.append(verb)
        verb_idx = len(entities) - 1
        for arg_annotation in arg_annotations:
            entities.append(arg_annotation)
            relations.append((verb_idx, len(entities) - 1, arg_annotation[2]))
    return entities, relations


def merge_two_annotations(first: List[List[str]], second: List[List[str]],
                          generator_func=bio_generator) -> List[List[str]]:
    """
    Merges two annotations in BIO format.

    Gives priority to the first annotation.

    Deep copies the result.

    Examples:
        `[B-x, I-x, O, O]` and `[O, B-y, I-y, O]` will be merged as `[B-x, I-x, O, O]`
        `[O, B-x, I-x, O]` and `[B-y, I-y, O, O]` will be merged as `[O, B-x, I-x, O]`
    Args:
        first:
        second:

    Returns:

    """
    # sanity check
    assert len(first) == len(second)
    result: List[List[str]] = []
    for i, sentence_first in enumerate(first):
        sentence_second = second[i]
        sentence_result: List[str] = sentence_first[:]
        # sanity check
        assert len(sentence_first) == len(sentence_second)
        for (start, end), _ in generator_func(sentence_second):
            if all(t == 'O' for t in sentence_result[start:end + 1]):
                sentence_result[start:end + 1] = sentence_second[start:end + 1]
        result.append(sentence_result)
    return result


def get_all_subclasses(cls):
    """
    Returns all (currently imported) subclasses of a given class.

    :param cls: Class to get subclasses of.

    :return: all currently imported subclasses.
    """
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in get_all_subclasses(c))


# def failsafe_combined_rule_sentence_segmenter(doc: Doc):
#     if doc:
#         return combined_rule_sentence_segmenter(doc)
#     else:
#         return doc


class OneSentSentencizer:
    name = "one-sent-sentencizer"

    def __call__(self, doc: Doc):
        for i in range(len(doc)):
            doc[i].is_sent_start = False
        if doc:
            doc[0].is_sent_start = True
        return doc


def convert_clusters_to_offsets(doc: Doc, clusters: List[List[Tuple[int, int]]]) -> CharOffsetAnnotation:
    result = []
    for i, cluster in enumerate(clusters):
        logger.trace(cluster)
        logger.trace(doc)
        for first_token, last_token in cluster:
            last_token = last_token
            logger.debug(f"doc[{first_token}:{last_token + 1}] = {doc[first_token:last_token + 1]}")
            start = doc[first_token].idx
            end = doc[last_token].idx + len(doc[last_token])
            logger.debug(f"doc.char_span[{start}:{end}] = {doc._.char_span_relaxed(start, end)}")
            result.append((start, end, f"CLUSTER-{i}"))
    return result
