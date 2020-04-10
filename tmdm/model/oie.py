from collections import defaultdict
from typing import List, Tuple, Dict

from loguru import logger
from spacy.tokens.doc import Doc

import tmdm.model
from tmdm.model.extensions import Annotation, extend


def set_oies(self: Doc, oies):
    self._._oies = oies


@extend(Doc, 'property', create_attribute=True, default=[], setter=set_oies)
def oie(self: Doc) -> List['Verb']:
    # oie_extractions = []
    # if not Verb.cache[self._.name]:
    result = []
    tags: List[List[Tuple[int, int, str]]] = self._._oies
    if not tags:
        logger.warning("No OIEs extracted for this document (yet?).")
        logger.debug(tags)
    for i, verb_tags in enumerate(tags):
        logger.debug("Beginning...")
        verb_start, verb_end = next((s, e) for s, e, l in verb_tags if l == "VERB")
        span = self.char_span(verb_start, verb_end)
        verb_start, verb_end = span.start, span.end
        verb = Verb(self, verb_start, verb_end, "VERB")
        logger.info(f"Processing: {verb}")
        verb.update_token_annotations(i)
        args = []
        logger.debug(f"Processed {verb}")
        for j, (arg_start, arg_end, order) in enumerate(((s, e, l) for s, e, l in verb_tags if not l == 'VERB'),
                                                        len(Argument.cache[tmdm.model.extensions.name])):
            # TODO: potentially: split order by '-'
            span = self.char_span(arg_start, arg_end)
            arg_start, arg_end = span.start, span.end
            a = Argument(self, arg_start, arg_end, order)
            a.update_token_annotations(j)
            a.verb = verb
            args.append(a)

        verb.set_arguments(args)
        all_spans = sorted([(verb_start, verb_end)] + [(a.start, a.end) for a in args])

        # explosion of pythonicism
        verb.continuous = all(
            end_previous == start_next for (_, end_previous), (start_next, _)
            in zip(all_spans, all_spans[1:])
        )
        verb.extended_span = all_spans[0][0], all_spans[-1][1]
    return result
    # TODO: cache later
    # Verb.cache[self._.name].append(verb)
    # Argument.cache[self._.name].extend(args)


class Verb(Annotation):
    arguments: List['arguments']
    arguments_as_map: Dict[int, 'Argument']
    extended_span: Tuple[int, int]
    continuous: bool

    def set_arguments(self, arguments: List['Argument']):
        self.arguments = arguments
        self.arguments_as_map = {a.label_: a for a in arguments}

    cache = defaultdict(list)

    @classmethod
    def make(cls, doc: Doc, idx: int, start, end, label, *args, **kwargs):
        ...


# else:
# logger.debug(f"lazily loaded already...")
# return Verb.cache[self._.name]


# verb, argument_spans = extract_oie_spans(verb['tags'])
#
# if verb:
#     verb = document.convert_relative_absolute(
#         sentence_num, verb)
#     predicate = Predicate(document, verb, sentence_num, [])
#     args = [
#         Argument(
#             document, sentence_num, predicate,
#             document.convert_relative_absolute(
#                 sentence_num, span), order)
#         for span, order in argument_spans
#     ]
#     predicate.set_arguments(args)
#     all_spans = sorted([predicate.span] +
#                        [a.span for a in predicate.arguments])
#     predicate.continuous = all(
#         all_spans[i][1] == all_spans[i + 1][0]
#         for i in range(len(all_spans) - 1))
#     predicate.extended_span = (all_spans[0][0],
#                                all_spans[-1][1])
#     oie_extractions.append(predicate)

# Prune same arguments
# Argument.merge_all(arg for p in oie_extractions for arg in p.arguments)
# return oie_extractions


class Argument(Annotation):
    verb: Verb

    @classmethod
    def make(cls, **kwargs):
        ...
