from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc
from tmdm.classes import CharOffsetAnnotation, Provider
from transformers import LukeTokenizer, LukeForEntityPairClassification
from tmdm.pipe.pipe import PipeElement


class OnlineRCProvider(Provider):
    name = 'transformers-luke-rc-provider'

    def __init__(self, cuda=-1):
        self.model = None
        self.tokenizer = None
        self.cuda = cuda # Not used
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

    def postprocess_document(self, results, all_ents):
        # Set to 'no_relation' if prediction ambiguous
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                if predictions[1] - predictions[3] < 0.4:
                    results[id1][id2] = -1

        # Set to 'no_relation' if arguments are of wrong type
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                if predictions == -1:
                    continue

                pred = predictions[0]
                if pred < 17 and all_ents[id1].label_ != "ORG":
                    results[id1][id2] = -1
                    continue
                if pred >= 17 and all_ents[id1].label_ != "PER":
                    results[id1][id2] = -1
                    continue
                if pred in [4,5,28,29] and all_ents[id2].label_ != "DATE":
                    results[id1][id2] = -1
                    continue
                if pred in [10,12,15,18,21,32,33,36,37] and all_ents[id2].label_ != "PER":
                    results[id1][id2] = -1
                    continue
                if pred in [2,3,13,22,23,24,25,26,27,38,39,40] and all_ents[id2].label_ != "LOC":
                    results[id1][id2] = -1
                    continue

        # Enforce symmetry of some relations
        for id1, relations in results.items():
            for id2, predictions in relations.items():
                if predictions == -1:
                    continue

                if predictions[0] == 21:
                    if id2 not in results:
                        results[id2] = {}
                    results[id2][id1] = [33, -1, -1, -1]
                elif predictions[0] == 33:
                    if id2 not in results:
                        results[id2] = {}
                    results[id2][id1] = [21, -1, -1, -1]
                elif predictions[0] == 32:
                    if id2 not in results:
                        results[id2] = {}
                    results[id2][id1] = [32, -1, -1, -1]
                elif predictions[0] == 36:
                    if id2 not in results:
                        results[id2] = {}
                    results[id2][id1] = [36, -1, -1, -1]
                elif predictions[0] == 37:
                    if id2 not in results:
                        results[id2] = {}
                    results[id2][id1] = [37, -1, -1, -1]

        return results

    def annotate_document(self, doc: Doc):
        all_ents = doc._.nes # TODO: Add coreferences to this

        results = {}
        text = doc.text
        for i in range(len(all_ents)):
            for j in range(len(all_ents)):
                ent1 = all_ents[i]
                ent2 = all_ents[j]

                # TODO: Also don't run if in eachothers clusters maybe
                if i == j or (ent1.label_ != "PER" and ent1.label_ != "ORG"):
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

        results = self.postprocess_document(results, all_ents)
        return self.convert(results, all_ents)


def get_rc_pipe(cuda=-1):
    return PipeElement(name='rc', field='relations',provider=OnlineRCProvider(cuda=cuda))
