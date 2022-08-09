from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc

from tmdm.classes import CharOffsetAnnotation
from tmdm.pipe.pipe import PipeElement
from tmdm.transformers.common import OnlineProvider
from tmdm.util import get_offsets_from_sentences
from transformers.pipelines import pipeline

DocumentLevelTransformerEntitiesAnnotation = List[List[Dict[str, Any]]]


def convert(doc: Doc, result: DocumentLevelTransformerEntitiesAnnotation, filter_subwords=True) -> CharOffsetAnnotation:
    results = []
    assert len(list(doc.sents)) == len(result)

    for sent, ann in zip(doc.sents, result):
        sent_offset = sent.start_char
        for s, e, t, w in [(e['start'], e['end'], e['entity_group'], e['word']) for e in ann]:
            if filter_subwords and w.startswith('##'):
                pass
            else:
                results.append((sent_offset + s, sent_offset + e, t))
    return results


class OnlineNerProvider(OnlineProvider):
    def __init__(self, task: str, post_process, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        self.post_process = post_process

    def load(self, _=None):
        self.pipeline = pipeline(self.task, model=self.path_or_name, tokenizer=self.path_or_name_tokenizer, device=self.cuda)# aggregation_strategy='first')

    def annotate_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        logger.trace("Entering annotate batch...")
        docs = list(docs)
        instances = [self.preprocess(doc) for doc in docs] if self.preprocess else docs
        # try:
        flat_instances = [i for l in instances for i in l]
        if not flat_instances:
            logger.debug("Everything is empty!")
            return [[] for _ in docs]
        logger.debug(flat_instances)
        result = self.pipeline(flat_instances)
        if not result:
            logger.debug("No named entities recognized!")
            return [[] for _ in docs]
        if result and not isinstance(result[0], list):
            result = [result]
        logger.debug(f"Result: {result}")
        # except Exception as e:
        #    logger.error(str(e))
        #    return [([], []) for _ in docs]
        result_iterator = iter(result)
        # batched_results = [[self.pipeline.group_entities(next(result_iterator)) for _ in d.sents] for d in docs]
        batched_results = [[self.pipeline.group_entities(next(result_iterator)) for _ in d.sents] for d in docs]
        #batched_results = [[next(result_iterator) for _ in d.sents] for d in docs]
        logger.debug(batched_results)
        batched_results_iter = iter(batched_results)
        return [self.converter(doc, next(batched_results_iter)) for doc in docs]

    def postprocess_batch(self, docs: List[Doc]) -> List[CharOffsetAnnotation]:
        alltuples = []
        for doc in docs:
            doctuples = []
            for ne in doc._.nes:
                doctuples.append([ne.start_char, ne.end_char, ne.label_])

            # Merge until cannot merge anymore
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

            for i in range(len(doctuples)):
                doctuples[i] = tuple(doctuples[i])
            alltuples.append(doctuples)

        return alltuples


def get_ne_pipe(post_process: bool = False, model: str = None, tokenizer: str = None, cuda=-1):
    return PipeElement(name='ner', field='nes',
                       provider=OnlineNerProvider(task="ner", path_or_name=model,
                                                  path_or_name_tokenizer=tokenizer,
                                                  converter=convert, cuda=cuda, post_process=post_process))
