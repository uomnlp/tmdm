from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp_models.syntax.srl.openie_predictor import sanitize_label, consolidate_predictions
from collections import defaultdict
from loguru import logger
from spacy.tokens import Doc, Token, Span
from typing import List, Dict, Callable
import numpy as np
from tmdm.allennlp.common import OnlineProvider
from tmdm.classes import OffsetAnnotation
from allennlp_models.syntax.srl import OpenIePredictor
from spacy.gold import offsets_from_biluo_tags, iob_to_biluo

from tmdm.util import entities_relations_from_by_verb

ModelOutput = Dict[str, np.ndarray]


# def join_mwp(tags: List[str]) -> List[str]:
#     """
#     Join multi-word predicates to a single
#     predicate ('V') token.
#     """
#     ret = []
#     verb_flag = False
#     for tag in tags:
#         if "V" in tag:
#             # Create a continuous 'V' BIO span
#             prefix, _ = tag.split("-", 1)
#             if verb_flag:
#                 # Continue a verb label across the different predicate parts
#                 prefix = "I"
#             ret.append(f"{prefix}-V")
#             verb_flag = True
#         else:
#             ret.append(tag)
#             verb_flag = False
#
#     return ret


def join_mwp(tags: List[str], mask: List[int]):
    # assuming that the predicate is continuous
    predicate_start = next(i for i, is_predicate in enumerate(mask) if is_predicate)
    logger.trace(f"Predicate start: {predicate_start}")
    predicate_length = sum(mask)
    pre_verb = tags[:predicate_start]
    verb = ['B-V'] + ["I-V"] * (predicate_length - 1)
    one_past_verb = tags[predicate_start + predicate_length]
    if one_past_verb.startswith("I"):
        _, ann = one_past_verb.split("-")
        new_one_past_verb = f"B-{ann}"
    else:
        new_one_past_verb = one_past_verb
    rest = tags[predicate_start + predicate_length + 1:]
    logger.trace(f"MWP joined: {pre_verb} + {verb} + {[new_one_past_verb]} + {rest}")
    return pre_verb + verb + [new_one_past_verb] + rest


def _extract_predicates(sent):
    pred_ids = []
    for i, t in enumerate(sent):
        if t.pos_ == "VERB" or (t.pos_ == "PART" and i > 0 and sent[i - 1].pos_ == "VERB"):
            pred_ids.append(i)
    masks = [[0] * len(sent)]
    masks[0][pred_ids[0]] = 1
    for (prev_idx, idx) in zip(pred_ids, pred_ids[1:]):
        # consecutive
        if idx - prev_idx == 1:
            masks[len(masks) - 1][idx] = 1
        else:
            mask = [[0] * len(sent)]
            mask[idx] = 1
            masks.append(mask)
    return masks


def post_process(results: List[ModelOutput], sent_tokens: List[Token], predicate_mask: List[int]):
    outputs: List[List[str]] = [[sanitize_label(label) for label in result['tags']] for result in results]
    logger.trace(f"Outputs: {outputs}")
    pred_dict = consolidate_predictions(outputs, sent_tokens)
    logger.trace(f"Consolidated predictions: {pred_dict}")
    # Build and return output dictionary
    final_results: JsonDict = {"verbs": []}

    for tags in pred_dict.values():
        # Join multi-word predicates
        tags = join_mwp(tags, predicate_mask)

        # Create description text
        # description = make_oie_string(sent_tokens, tags)

        # Add a predicate prediction to the return dictionary.
        final_results["verbs"].append(tags)
    logger.trace(f"Final results: {final_results}")
    return sanitize(final_results)


def _extract_predicates_simple(sent: Span) -> List[List[int]]:
    verb_indices = [i for (i, t)
                    in enumerate(sent)
                    if t.pos_ == "VERB"]
    masks = [[0] * len(sent) for _ in verb_indices]

    for i, mask in enumerate(masks):
        mask[verb_indices[i]] = 1
    return masks


@Predictor.register("open-information-extraction", exist_ok=True)
class CustomOpenIEPredictor(OpenIePredictor):
    extract_predicates: Callable[[Span], List[List[int]]]

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        # noinspection PyTypeChecker
        return self.predict_batch_documents(instances)

    def predict_batch_documents(self, inputs: List[Doc]):
        logger.trace(f"Docs: {inputs}")
        instances: Dict[Doc, List[List[Instance]]] = defaultdict(list)
        for d in inputs:
            logger.trace(f"Doc: {d}")
            # TODO: make smarter predicate selection
            for sent_nr, sentence in enumerate(d.sents):
                # logger.trace(f"{sent_nr}: {sentence}")
                # verb_indices = [i for (i, t)
                #                 in enumerate(sentence)
                #                 if t.pos_ == "VERB"]
                masks = self.extract_predicates(sentence)

                # for i, mask in enumerate(masks):
                #     mask[verb_indices[i]] = 1
                logger.trace(f"masks: {masks}")
                instances[d].append([self._dataset_reader.text_to_instance(sentence, mask) for mask in masks])
        logger.trace(f"Instances: {instances}")
        # tokens.append(sentence)
        try:
            results = self._model.forward_on_instances(
                [inst for _, sents in instances.items() for sent in sents for inst in sent])
        except Exception as e:
            logger.error(str(e))
            return [[] for _ in inputs]
        results_iterator = iter(results)
        masks_iter = iter(masks)
        # per doc/ per sent / per verb / per token
        gathered_results: List[List[List[List[str]]]] = list()
        for document, per_predicate in instances.items():
            doc_sents = list(document.sents)
            per_doc = []
            for i, sentence in enumerate(per_predicate):
                r = [next(results_iterator) for _ in sentence]
                per_doc.append(post_process(r, doc_sents[i], next(masks_iter))['verbs'])
            gathered_results.append(per_doc)
        return gathered_results


def _convert_annotations(doc: Doc, annotations: List[List[List[str]]]) -> OffsetAnnotation:
    # verbs: Per sent, per verb per token
    result = []
    for sent, sent_annotations in zip(doc.sents, annotations):
        for per_verb in sent_annotations:
            r = offsets_from_biluo_tags(sent.as_doc(), iob_to_biluo(per_verb))
            sent_offset = sent[0].idx
            r = [(start + sent_offset, end + sent_offset, label) for start, end, label in r]
            result.append(r)

    return entities_relations_from_by_verb(result)


def get_oie_provider(model_path: str, extract_predicates: Callable[[Span], List[List[int]]] = None):
    extract_predicates = extract_predicates or _extract_predicates
    p = OnlineProvider(task='open-information-extraction', path=model_path, converter=_convert_annotations,
                       preprocessor=None)
    p.predictor.extract_predicates = extract_predicates
    return p
