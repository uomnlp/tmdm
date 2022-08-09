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

    def annotate_document(self, doc: Doc):
        relation_idx = 0
        results = []
        text = doc.text
        all_ents = doc._.nes # TODO: Add coreferences to this
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
                pred_idx = int(outputs.logits[0].argmax())
                if pred_idx != 0:
                    pred_relation = self.model.config.id2label[pred_idx]
                    subjlabel = str(relation_idx) + "-subj-" + pred_relation
                    results.append((ent1.start_char, ent1.end_char, subjlabel))
                    objlabel = str(relation_idx) + "-obj-" + pred_relation
                    results.append((ent2.start_char, ent2.end_char, objlabel))
                    relation_idx += 1

        return results


def get_rc_pipe(cuda=-1):
    return PipeElement(name='rc', field='relations',provider=OnlineRCProvider(cuda=cuda))
