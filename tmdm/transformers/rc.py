from typing import Any, Dict, List, Tuple
from loguru import logger
from copy import deepcopy
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
        self.cuda = cuda # Not used
        self.model = None
        self.tokenizer = None
        self.load()

    def save(self, path: str):
        pass

    def load(self):
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

    def convert(self, results, all_ents):
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

    def annotate_document(self, doc: Doc):
        cluster_types = {}
        if self.with_coref:
            all_ents = doc._.corefs
            for nes in doc._.nes:
                skip = False
                for coref in doc._.corefs:
                    if overlaps_or_overlapped_by(coref, nes):
                        cluster_types[coref.label_] = nes.label_
                        skip = True
                        break
                if not skip:
                    all_ents.append(nes)
        else:
            all_ents = doc._.nes

        results = {}
        text = doc.text
        for i in range(len(all_ents)):
            for j in range(len(all_ents)):
                ent1 = all_ents[i]
                ent2 = all_ents[j]

                if i == j or (ent1.label_.startswith("CLUSTER") and ent1.label_ == ent2.label_):
                    continue

                ent1_type = get_nes_or_coref_type(ent1, cluster_types)
                if ent1_type != "PER" and ent1_type != "ORG":
                    continue

                entity_spans = [(ent1.start_char, ent1.end_char), (ent2.start_char, ent2.end_char)]
                inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
                outputs = self.model(**inputs)

                # Best prediction
                logits = outputs.logits
                top_pred_idx = int(logits[0].argmax())
                top_confidence = float(logits[0][top_pred_idx])

                if top_pred_idx == 0:
                    continue

                # 2nd best prediction
                logits[0][top_pred_idx] = 0
                second_pred_idx = int(logits[0].argmax())
                second_confidence = float(logits[0][second_pred_idx])

                if i not in results:
                    results[i] = {}
                results[i][j] = [top_pred_idx, top_confidence, second_pred_idx, second_confidence]

        results = self.postprocess_document(results, all_ents, cluster_types)
        return self.convert(results, all_ents)


def get_rc_pipe(with_coref=False, cuda=-1):
    return PipeElement(name='rc', field='relations',provider=OnlineRCProvider(with_coref=with_coref, cuda=cuda))
