from typing import Any, Dict, List, Tuple
from loguru import logger
from copy import deepcopy
from torch import cat
from spacy.tokens import Doc
from tmdm.classes import CharOffsetAnnotation, Provider
from transformers import LukeTokenizer, LukeForEntityPairClassification
from tmdm.pipe.pipe import PipeElement


def overlaps_or_overlapped_by(ent1, ent2):
    fully_overlaps = (ent1.start_char <= ent2.start_char and ent2.end_char <= ent1.end_char)
    is_fully_overlaped = (ent2.start_char < ent1.start_char  and ent1.end_char < ent2.end_char)
    return fully_overlaps or is_fully_overlaped


def get_nes_or_coref_type(ent, cluster_types):
    if ent.label_.startswith("CLUSTER"):
        if ent.label_ in cluster_types:
            return cluster_types[ent.label_]
        else:
            return "MISC"
    else:
        return ent.label_


class OnlineRCProvider(Provider):
    name = 'transformers-luke-rc-provider'

    def __init__(self, with_coref=False, cuda=-1):
        self.with_coref = with_coref
        self.cuda = cuda
        self.model = None
        self.tokenizer = None
        self.load()

    def save(self, path: str):
        pass

    def load(self):
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
        if self.cuda == 0:
            self.model.to("cuda")
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

    def convert(self, results, all_ents):
        # Format is two tuples per relation: [(<ent1 start>, <ent1 end>, "<relation id>-subj-<relation type>"),
        #                                     (<ent2 start>, <ent2 end>, "<relation id>-obj-<relation type>"), ...]
        relation_idx = 0
        converted = []
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                if predictions == -1:
                    continue

                pred_relation = self.model.config.id2label[predictions[0]]
                subjlabel = str(relation_idx) + "-subj-" + pred_relation
                converted.append((all_ents[id1].start_char, all_ents[id1].end_char, subjlabel))
                objlabel = str(relation_idx) + "-obj-" + pred_relation
                converted.append((all_ents[id2].start_char, all_ents[id2].end_char, objlabel))
                relation_idx += 1

        return converted

    def postprocess_document(self, results, all_ents, cluster_types):
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                # Set to 'no_relation' if prediction ambiguous
                if predictions[1] - predictions[3] < 0.4:
                    results[id1][id2] = -1
                    continue

                # Set to 'no_relation' if arguments are of wrong type
                pred = predictions[0]

                ent1_type = get_nes_or_coref_type(all_ents[id1], cluster_types)
                if pred < 17 and ent1_type != "ORG":
                    results[id1][id2] = -1
                    continue
                if pred >= 17 and ent1_type != "PER":
                    results[id1][id2] = -1
                    continue

                ent2_type = get_nes_or_coref_type(all_ents[id2], cluster_types)
                if pred in [4,5,28,29] and ent2_type != "DATE":
                    results[id1][id2] = -1
                    continue
                if pred in [10,12,15,18,21,32,33,36,37] and ent2_type != "PER":
                    results[id1][id2] = -1
                    continue
                if pred in [2,3,13,22,23,24,25,26,27,38,39,40] and ent2_type != "LOC":
                    results[id1][id2] = -1
                    continue

        # Enforce symmetry of some relations
        results_copy = deepcopy(results)
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                if predictions == -1:
                    continue

                if predictions[0] == 21:
                    if id2 not in results_copy:
                        results_copy[id2] = {}
                    results_copy[id2][id1] = [33, -1, -1, -1]
                elif predictions[0] == 33:
                    if id2 not in results_copy:
                        results_copy[id2] = {}
                    results_copy[id2][id1] = [21, -1, -1, -1]
                elif predictions[0] == 32:
                    if id2 not in results_copy:
                        results_copy[id2] = {}
                    results_copy[id2][id1] = [32, -1, -1, -1]
                elif predictions[0] == 36:
                    if id2 not in results_copy:
                        results_copy[id2] = {}
                    results_copy[id2][id1] = [36, -1, -1, -1]
                elif predictions[0] == 37:
                    if id2 not in results_copy:
                        results_copy[id2] = {}
                    results_copy[id2][id1] = [37, -1, -1, -1]

        #TODO: Also enforce transitive types? member of member, family of family
        return results_copy

    def try_to_run_model(self, texts, entity_spans):
        try:
            inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
            if self.cuda == 0:
                inputs = inputs.to("cuda")
            outputs = self.model(**inputs)
            return outputs.logits
        except RuntimeError:
            logger.info("Ran out of memory for RC, halving workload...")
            half = len(texts) // 2
            first_logits = self.try_to_run_model(texts[:half], entity_spans[:half])
            second_logits = self.try_to_run_model(texts[half:], entity_spans[half:])
            return cat((first_logits, second_logits))

    def annotate_document(self, doc: Doc) -> CharOffsetAnnotation:
        return self.annotate_batch([doc])[0]

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        # Get all ents in docs (with or without corefs)
        docs_all_ents = []
        docs_cluster_types = []
        for d in range(len(docs)):
            cluster_types = {}
            if self.with_coref:
                all_ents = docs[d]._.corefs
                for nes in docs[d]._.nes:
                    skip = False
                    for coref in docs[d]._.corefs:
                        # If NE in coref cluster then cluster takes NEs type
                        if overlaps_or_overlapped_by(coref, nes):
                            cluster_types[coref.label_] = nes.label_
                            skip = True
                            break
                    # Don't include NE that is the same as a coref already included
                    if not skip:
                        all_ents.append(nes)
            else:
                all_ents = docs[d]._.nes

            docs_all_ents.append(all_ents)
            docs_cluster_types.append(cluster_types)

        # Get viable pairs of ents to classify
        texts = []
        entity_spans = []
        entity_combos = []
        for d in range(len(docs)):
            doc_text = docs[d].text
            all_ents = docs_all_ents[d]
            for i in range(len(all_ents)):
                for j in range(len(all_ents)):
                    ent1 = all_ents[i]
                    ent2 = all_ents[j]

                    # Not viable if ents are eachother or from same cluster
                    if i == j or (ent1.label_.startswith("CLUSTER") and ent1.label_ == ent2.label_):
                        continue

                    # Not viable unless subject is Person or Organisation (TACRED classes)
                    ent1_type = get_nes_or_coref_type(ent1, docs_cluster_types[d])
                    if ent1_type != "PER" and ent1_type != "ORG":
                        continue

                    # Not viable if ents from far apart sentences
                    same = ent1.sent.start_char == ent2.sent.start_char and ent1.sent.end_char == ent2.sent.end_char
                    adjacent = (ent2.sent.end_char+1) == ent1.sent.start_char or (ent1.sent.end_char+1) == ent2.sent.start_char
                    if (not same) and (not adjacent):
                        continue

                    # Ents can spill over sentence ends so need to extend to include
                    text_start = min(ent1.sent.start_char, ent2.sent.start_char)
                    text_end = max([ent1.end_char, ent1.sent.end_char, ent2.end_char, ent2.sent.end_char])
                    text = doc_text[text_start:text_end]
                    texts.append(text)

                    span1 = (ent1.start_char - text_start, ent1.end_char - text_start)
                    span2 = (ent2.start_char - text_start, ent2.end_char - text_start)
                    entity_spans.append([span1, span2])

                    entity_combos.append([d, i, j])

        if len(entity_spans) == 0:
            return [[] for i in range(len(docs))]

        # Get the likeliness of each class for every pair
        logits = self.try_to_run_model(texts, entity_spans)

        # Get best predictions
        top_pred_idxs = logits.argmax(-1)
        top_confidences = logits.max(-1).values

        # Get 2nd best predictions
        for i in range(len(logits)):
            logits[i][top_pred_idxs[i]] = 0
        second_pred_idxs = logits.argmax(-1)
        second_confidences = logits.max(-1).values

        batch_results = []
        for d in range(len(docs)):
            # Get in format for post processing
            results = {}
            for idx, c in enumerate(entity_combos):
                if c[0] == d:
                    if c[1] not in results:
                        results[c[1]] = {}
                    results[c[1]][c[2]] = [int(top_pred_idxs[idx]), float(top_confidences[idx]),
                                           int(second_pred_idxs[idx]), float(second_confidences[idx])]
        
            results = self.postprocess_document(results, docs_all_ents[d], docs_cluster_types[d])
            results = self.convert(results, docs_all_ents[d])
            batch_results.append(results)

        return batch_results


def get_rc_pipe(with_coref=False, cuda=-1):
    return PipeElement(name='rc', field='relations',provider=OnlineRCProvider(with_coref=with_coref, cuda=cuda))
