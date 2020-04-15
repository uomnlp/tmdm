from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp_models.syntax.srl.openie_predictor import sanitize_label, consolidate_predictions
from allennlp_models.syntax.srl.openie_predictor import join_mwp as allennlp_join_mwp
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


def my_join_mwp(tags: List[str], mask: List[int]):
    assert len(tags) == len(mask), f"len(tags) != len(mask) ({len(tags)} != {len(mask)})"
    # assuming that the predicate is continuous
    predicate_start = next(i for i, is_predicate in enumerate(mask) if is_predicate)
    logger.trace(f"Predicate start: {predicate_start}")
    predicate_length = sum(mask)
    if not predicate_length:
        return tags
    pre_verb = tags[:predicate_start]
    verb = ['B-V'] + ["I-V"] * (predicate_length - 1)
    new_one_past_verb = []
    if predicate_start + predicate_length < len(tags):
        one_past_verb = tags[predicate_start + predicate_length]
        if one_past_verb.startswith("I"):
            _, ann = one_past_verb.split("-", 1)
            new_one_past_verb = [f"B-{ann}"]
        else:
            new_one_past_verb = [one_past_verb]
    rest = tags[predicate_start + predicate_length + 1:]
    logger.trace(f"MWP joined: {pre_verb} + {verb} + {new_one_past_verb} + {rest}")
    result = pre_verb + verb + new_one_past_verb + rest
    logger.trace(f"Result: {result}")
    logger.trace(f"mask: {mask}")
    logger.trace(f"tags: {tags}")
    assert len(result) == len(tags)
    return result


def _extract_predicates(sent):
    pred_ids = []
    for i, t in enumerate(sent):
        if t.pos_ == "VERB" or (t.pos_ == "PART" and i > 0 and sent[i - 1].pos_ == "VERB"):
            pred_ids.append(i)
    masks = [[0] * len(sent)]
    if not pred_ids:
        return masks
    masks[0][pred_ids[0]] = 1
    for (prev_idx, idx) in zip(pred_ids, pred_ids[1:]):
        assert idx < len(sent)
        # consecutive
        if idx - prev_idx == 1:
            masks[len(masks) - 1][idx] = 1
        else:
            mask = [0] * len(sent)
            mask[idx] = 1
            masks.append(mask)
    return masks


def post_process(results: List[ModelOutput], sent_tokens: List[Token], predicate_mask: List[List[int]] = None):
    outputs: List[List[str]] = [[sanitize_label(label) for label in result['tags']] for result in results]
    logger.trace(f"Outputs: {outputs}")
    pred_dict = consolidate_predictions(outputs, sent_tokens)
    logger.trace(f"Consolidated predictions: {pred_dict}")
    # Build and return output dictionary
    final_results: JsonDict = {"verbs": []}
    predicate_mask = predicate_mask or [None for _ in pred_dict.values()]
    for tags, mask in zip(pred_dict.values(), predicate_mask):
        if mask:
            # Join multi-word predicates
            tags = my_join_mwp(tags, mask)
        else:
            tags = allennlp_join_mwp(tags)
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
    simple_predicates: bool

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        # noinspection PyTypeChecker
        return self.predict_batch_documents(instances)

    def predict_batch_documents(self, inputs: List[Doc]):
        logger.trace(f"Docs: {inputs}")
        instances: Dict[Doc, List[List[Instance]]] = defaultdict(list)
        # all_masks: Dict[Doc, List[List[List[int]]]] = defaultdict(list)
        all_masks = []
        for d in inputs:
            if d:
                logger.trace(f"Doc: {d}")
                for sent_nr, sentence in enumerate(d.sents):
                    if self.simple_predicates:
                        masks = _extract_predicates_simple(sentence)
                    else:
                        masks = _extract_predicates(sentence)
                    masks = [m for m in masks if sum(m) > 0]
                    instances[d].append([self._dataset_reader.text_to_instance(sentence, mask) for mask in masks])
                    all_masks.extend(masks)
            else:
                instances[d].append([])
        logger.trace(f"Instances: {instances}")
        # tokens.append(sentence)
        # try:
        flat_instances = [inst for _, sents in instances.items() for sent in sents for inst in sent]
        if flat_instances:
            results = self._model.forward_on_instances(flat_instances)
        else:
            results = [list() for _ in instances.keys()]
            logger.trace(f"Whole batch is empty... {results}")
            return results
        assert len(results) == len(all_masks)
        # except Exception as e:
        #    logger.error(str(e))
        #    return [([],[]) for _ in inputs]
        results_iterator = iter(results)
        masks_iter = iter(all_masks)
        # per doc/ per sent / per verb / per token
        gathered_results: List[List[List[List[str]]]] = list()
        for document, per_predicate in instances.items():
            if document:
                doc_sents = list(document.sents)
                per_doc = []
                for i, sentence in enumerate(per_predicate):
                    r = [next(results_iterator) for _ in sentence]
                    m = [next(masks_iter) for _ in sentence]
                    if self.simple_predicates:
                        per_doc.append(post_process(r, doc_sents[i])['verbs'])
                    else:
                        per_doc.append(post_process(r, doc_sents[i], m)['verbs'])

                gathered_results.append(per_doc)
            else:
                gathered_results.append([])
        return gathered_results


def _convert_annotations(doc: Doc, annotations: List[List[List[str]]]) -> OffsetAnnotation:
    # verbs: Per sent, per verb per token
    result = []
    for sent, sent_annotations in zip(doc.sents, annotations):
        for per_verb in sent_annotations:
            try:
                r = offsets_from_biluo_tags(sent.as_doc(), iob_to_biluo(per_verb))
            except IndexError:
                logger.error(f"{len(sent)} tokens in sent, {len(per_verb)} in annotation")
                raise ValueError(f"Cannot align {iob_to_biluo(per_verb)} for '{sent}'")
            sent_offset = sent[0].idx
            r = [(start + sent_offset, end + sent_offset, label) for start, end, label in r]
            result.append(r)

    return entities_relations_from_by_verb(result)


def get_oie_provider(model_path: str, simple_predicates: bool = False, cuda=-1):
    from allennlp_models.syntax.srl.srl_model import SemanticRoleLabeler
    p = OnlineProvider(task='open-information-extraction', path=model_path, converter=_convert_annotations,
                       preprocessor=None, cuda=cuda)
    p.predictor.simple_predicates = simple_predicates
    return p
