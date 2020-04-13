from functools import partial
from operator import attrgetter, itemgetter

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp_models.syntax import SrlReader
from allennlp_models.syntax.srl.openie_predictor import sanitize_label, consolidate_predictions, join_mwp, \
    make_oie_string, get_predicate_text
from collections import defaultdict
from loguru import logger
from overrides import overrides
from spacy.tokens import Doc, Token
from typing import List, Tuple, Dict, Any
import numpy as np
from tmdm.allennlp.common import OnlinePredictor
from tmdm.classes import OffsetAnnotation
from allennlp_models.syntax.srl import OpenIePredictor
from spacy.gold import offsets_from_biluo_tags, iob_to_biluo

from tmdm.util import entities_relations_from_by_verb

ModelOutput = Dict[str, np.ndarray]


def post_process(results: List[ModelOutput], sent_tokens: List[Token]):
    outputs: List[List[str]] = [[sanitize_label(label) for label in result['tags']] for result in results]

    pred_dict = consolidate_predictions(outputs, sent_tokens)

    # Build and return output dictionary
    final_results: JsonDict = {"verbs": []}

    for tags in pred_dict.values():
        # Join multi-word predicates
        tags = join_mwp(tags)

        # Create description text
        # description = make_oie_string(sent_tokens, tags)

        # Add a predicate prediction to the return dictionary.
        final_results["verbs"].append(tags)
    logger.trace(f"Final results: {final_results}")
    return sanitize(final_results)


@Predictor.register("open-information-extraction", exist_ok=True)
class CustomOpenIEPredictor(OpenIePredictor):
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        # noinspection PyTypeChecker
        return self.predict_batch_documents(instances)

    def predict_batch_documents(self, inputs: List[Doc]):
        instances: Dict[Doc, List[List[Instance]]] = defaultdict(list)
        for d in inputs:
            logger.trace(f"{d}")
            # TODO: make smarter predicate selection
            for sent_nr, sentence in enumerate(d.sents):
                logger.trace(f"{sent_nr}: {sentence}")
                verb_indices = [i for (i, t)
                                in enumerate(sentence)
                                if t.pos_ == "VERB"]
                masks = [[0] * len(sentence) for _ in verb_indices]

                for i, mask in enumerate(masks):
                    mask[verb_indices[i]] = 1
                logger.trace(f"masks: {masks}")
                instances[d].append([self._dataset_reader.text_to_instance(sentence, mask) for mask in masks])
        logger.debug(instances)
        # tokens.append(sentence)
        try:
            results = self._model.forward_on_instances(
                [inst for _, sents in instances.items() for sent in sents for inst in sent])
        except Exception as e:
            logger.error(str(e))
            return [[] for _ in inputs]
        results_iterator = iter(results)
        # per doc/ per sent / per verb / per token
        gathered_results: List[List[List[List[str]]]] = list()
        for document, sentences in instances.items():
            per_doc = []
            # gathered_results[document].append([])
            for i, sentence in enumerate(sentences):
                r = [next(results_iterator) for _ in sentence]
                # gathered_results[document][i].append(post_process(r, document._spacy_tokens[i])['verbs'])Cy
                per_doc.append(post_process(r, list(document.sents)[i])['verbs'])
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


# def _preprocess(dr: SrlReader, doc: Doc) -> List[Instance]:
#     instances = []
#     for sent in doc.sents:
#         for i, verb in enumerate(t for t in sent if is_verb(doc, t)):
#             tokens = [t for t in sent]
#             verb_mask = [0] * len(sent)
#             verb_mask[i] = 1
#             instances.append(dr.text_to_instance(tokens, verb_mask))
#     return instances


def get_oie_predictor(model_path: str):
    from allennlp_models.coref.coref_model import CoreferenceResolver
    converter = _convert_annotations
    # getter = itemgetter("verbs")
    p = OnlinePredictor(task='open-information-extraction', path=model_path, converter=converter)
    return p
