from typing import Any, Dict, List, Tuple
from loguru import logger
from spacy.tokens import Doc, Span

from tmdm.classes import CharOffsetAnnotation, Provider
from tmdm.pipe.pipe import PipeElement
from tmdm.util import get_offsets_from_sentences
from requests.exceptions import HTTPError

import spacy
import argparse
import json
import sys

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init

import tmdm.blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tmdm.blink.biencoder import BiEncoderRanker, load_biencoder
from tmdm.blink.data_process import (
    process_mention_data,
    get_candidate_representation,
)

import handystuff.loaders

def find_within(ents, start, end):
    for e in ents:
        logger.debug(f"({start},{end})  vs ({e[0]},{e[1]})")
        if e[0] >= start and e[1] <= end:
            yield e
            
def find_overlap(ents, start, end):
    for e in ents:
        if (e[0] >= start and e[0] <= end) or (e[1] >= start and e[1] <= end):
            return True
    return False
            

def annotate(ner_model, input_sentences, output_dates=False):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = str(mention["labels"][0])
        record["label_id"] = -1
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = int(mention["sent_idx"])
        samples.append(record)
    return samples

def select_meta(ent, rich):
    if rich:
        return {'uri': ent['URI'], 'support': ent['support'], 'types': ent['types'].split(',') if ent['types'] else [],
                'similarity': ent['similarityScore'], 'label': ent['URI']}
    else:
        return ent['URI']

def convert(doc: Doc, results, rich, nes_only) -> CharOffsetAnnotation:
    for ne, (s, e, l) in zip(doc._.nes, doc._._nes):
        overlap = False
        for r in results:
            if (r[0] >= s and r[0] <= e) or (r[1] >= s and r[1] <= e):
                overlap = True
                break
        if not overlap: results.append((s, e, l))
    return results

def load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )

def process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def run_biencoder(biencoder, dataloader, candidate_encoding, top_k=10, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in dataloader:
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores

class OnlineELProvider(Provider):
    name = 'blink-el-provider'
    
    def __init__(self, types=None, rich=False, nes_only=False, threshold=0.6, blink_folder="./models", with_date=False):
        self.types = types
        self.rich = rich
        self.nes_only = nes_only
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_lg") 
        self.blink_folder = blink_folder
        self.with_date = with_date
        self.load_models()

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass 
    
    def load_models(
            self,
    ):

        # load biencoder model
        with open(f"{self.blink_folder}/biencoder_wiki_large.json") as json_file:
            biencoder_params = json.load(json_file)
            biencoder_params["path_to_model"] = f"{self.blink_folder}/biencoder_wiki_large.bin"
        biencoder = load_biencoder(biencoder_params)

        crossencoder = None
        crossencoder_params = None

        # load candidate entities
        (
            candidate_encoding,
            title2id,
            id2title,
            id2text,
            wikipedia_id2local_id,
            faiss_indexer,
        ) = load_candidates(
            f"{self.blink_folder}/entity.jsonl", 
            f"{self.blink_folder}/all_entities_large.t7", 
            faiss_index=None, 
            index_path=None,
            logger=None,
        )
        
        self.biencoder = biencoder
        self.biencoder_params = biencoder_params
        self.crossencoder = crossencoder
        self.crossencoder_params = crossencoder_params
        self.candidate_encoding = candidate_encoding
        self.title2id = title2id
        self.id2title = id2title
        self.id2text = id2text
        self.wikipedia_id2local_id = wikipedia_id2local_id
        self.faiss_indexer = faiss_indexer

    def annotate_document(self, doc: Doc, threshold: int=0.9, top_k: int=10) -> CharOffsetAnnotation:
        return self.annotate_batch([doc], threshold, top_k)[0]
    
    def annotate_batch(self, docs: List[Doc], threshold: int=0.9, top_k: int=10) -> List[CharOffsetAnnotation]:
        
        docs = list(docs)
        id2url = {
            v: "https://en.wikipedia.org/wiki?curid=%s" % k
            for k, v in self.wikipedia_id2local_id.items()
        }
        predictions = []
        # Load NER model
        ner_model = NER.get_model(with_date=self.with_date)
        for doc in docs:
            text = str(doc)
            samples = annotate(ner_model, [text])
            # prepare the data for biencoder
            dataloader = process_biencoder_dataloader(
                samples, self.biencoder.tokenizer, self.biencoder_params
            )
            # run biencoder
            labels, nns, scores = run_biencoder(
                self.biencoder, dataloader, self.candidate_encoding, top_k, self.faiss_indexer
            )
            prediction = []
            idx = 0
            for entity_list, sample in zip(nns, samples):
                start = sample["start_pos"]
                end = sample["end_pos"]
                e_id = entity_list.tolist()[0]
                mention = sample["mention"].lower()
                title = self.id2title[e_id].lower()
                label = sample["label"].split()[0]
#                 similarity = self.nlp(mention).similarity(self.nlp(title))
                
                if label == "PER" and len(mention.split()) < 2: continue # Rule base filtering 1
#                 if similarity < threshold: continue # Rule base filtering 2
                url = id2url[e_id].split("?")[0]+"/"+self.id2title[e_id].replace(' ', '_') # reformat links to wiki pedia style
                info = {
                    'uri': url,
                    'support': 1,
                    'types': [],
                    'similarity': scores[idx].tolist()[0],
                    'label': sample["label"].split()[0]
                }
                prediction.append((start, end, info))
                idx += 1
            predictions.append(prediction)
        return [convert(doc, r, self.rich, self.nes_only) for doc, r in zip(docs, predictions)]

def get_blink_pipe(model: str = None, endpoint='http://kant.cs.man.ac.uk:2222/rest/annotate', rich=True, nes_only=False, threshold=0.9, blink_folder="./models", with_date=False):
    return PipeElement(name='el', field='nes', provider=OnlineELProvider(rich=rich, nes_only=nes_only, threshold=threshold, blink_folder=blink_folder, with_date=with_date))
