from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc

from tmdm.classes import CharOffsetAnnotation, Provider
from tmdm.pipe.pipe import PipeElement


class NEPostProcessProvider(Provider):
    name = 'ner-post-process-provider'

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def annotate_document(self, doc: Doc) -> CharOffsetAnnotation:
        return self.annotate_batch([doc])[0]

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        docs = list(docs)
        alltuples = []
        for doc in docs:
            doctuples = []
            for ne in doc._.nes:
                # Can split NEs in two if contain ' and ' but no longer optimal

                # if " and " in ne.text:
                #     and_start = ne.start_char + ne.text.index(" and ")
                #     and_end = and_start + len(" and ")
                #     doctuples.append([ne.start_char, and_start, ne.label_])
                #     doctuples.append([and_end, ne.end_char, ne.label_])
                # else:
                #     doctuples.append([ne.start_char, ne.end_char, ne.label_])

                doctuples.append([ne.start_char, ne.end_char, ne.label_])

            # Keep merging pairs of NEs if there are one or fewer characters inbetween until no more merges made
            changed = True
            while changed:
                changed = False
                for i in range(len(doctuples) - 1):
                    if doctuples[i] == None:
                        continue

                    end1 = doctuples[i][1]
                    start2 = doctuples[i + 1][0]
                    if end1 == start2 or (end1 + 1) == start2:
                        doctuples[i][1] = doctuples[i + 1][1]
                        changed = True
                        doctuples[i + 1] = None

                for i in range(len(doctuples) - 1, 0, -1):
                    if doctuples[i] == None:
                        del doctuples[i]

            # Convert back to NER tuple output format
            for i in range(len(doctuples)):
                doctuples[i] = tuple(doctuples[i])
            alltuples.append(doctuples)

        # Enable overwiting of cache since merging decreases total amount of NEs
        for doc in docs:
            for ne in doc._.nes:
                ne.cache.clear()

        return alltuples


def get_ne_post_process_pipe():
    return PipeElement(name='nepp', field='nes', provider=NEPostProcessProvider())
