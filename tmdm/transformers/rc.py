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
            text = docs[d].text
            all_ents = docs_all_ents[d]
            for i in range(len(all_ents)):
                for j in range(len(all_ents)):
                    ent1 = all_ents[i]
                    ent2 = all_ents[j]

                    # Not viable if ents are eachother or from same cluster
                    if i == j or (ent1.label_.startswith("CLUSTER") and ent1.label_ == ent2.label_):
                        continue

                    # Not viable unless subject is Person or Organisation
                    ent1_type = get_nes_or_coref_type(ent1, docs_cluster_types[d])
                    if ent1_type != "PER" and ent1_type != "ORG":
                        continue

                    entity_combos.append([d, i, j])
                    entity_spans.append([(ent1.start_char, ent1.end_char), (ent2.start_char, ent2.end_char)])
                    texts.append(text)

        if len(entity_spans) == 0:
            return [[] for i in range(len(docs))]

        inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        # Get best predictions
        logits = outputs.logits
        top_pred_idxs = logits.argmax(-1)
        top_confidences = logits.max(-1).values

        # Get 2nd best predictions
        for i in range(len(logits)):
            logits[i][top_pred_idxs[i]] = 0
        second_pred_idxs = logits.argmax(-1)
        second_confidences = logits.max(-1).values

        batch_results = []
        for d in range(len(docs)):
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
