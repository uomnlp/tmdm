
from typing import List

from spacy.tokens import Span

from tmdm.main import tmdm_pipeline
import tmdm.blink.ner as NER
from tmdm.blink.biencoder import BiEncoderRanker, load_biencoder
from tmdm.blink.el import process_biencoder_dataloader, annotate, load_candidates, run_biencoder
import json
import torch
import os.path

entity_catalogue = "./tmdm/models/entity.jsonl"
entity_encoding = "./tmdm/models/all_entities_large.t7"
blink_config = "./tmdm/models/biencoder_wiki_large.json"
blink_model = "./tmdm/models/biencoder_wiki_large.bin"

def test_model_exist():
    """
    Tests enitity and blink models exists in the directory
    """
    assert os.path.isfile(entity_catalogue)
    assert os.path.isfile(entity_encoding)
    assert os.path.isfile(blink_config)
    assert os.path.isfile(blink_model)

def test_blink_ner():
    input_sent = ["Bob ate a cheesecake."]
    ner_model = NER.get_model()
    prediction = ner_model.predict(input_sent)
    assert prediction["sentences"] == ['Bob ate a cheesecake.']
    assert prediction["mentions"][0]["text"] == "Bob"
    assert prediction["mentions"][0]["start_pos"] == 0
    assert prediction["mentions"][0]["end_pos"] == 3
    
    
def test_blink_annotate():
    """
    Tests blink's annotate function
    """
    input_sent = ["Bob ate a cheesecake."]
    ner_model = NER.get_model()
    samples = annotate(ner_model, ["Bob ate a cheesecake."])
    assert type(samples[0]) == dict, type(samples)
    assert samples[0]["label"].split()[0] == "PER"
    assert samples[0]["mention"] == "bob"
    
def test_blink_biencoder():
    """
    Tests blink's biencoder model is functional
    """
    with open(blink_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = blink_model
    biencoder = load_biencoder(biencoder_params)
    candidate_encoding = torch.zeros(100, 1024)
    samples = [{'label': 'PER (0.9705)', 'label_id': -1, 'context_left': '', 'context_right': ' ate a cheesecake.', 'mention': 'bob', 'start_pos': 0, 'end_pos': 3, 'sent_idx': 0}]
    dataloader = process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    labels, nns, scores = run_biencoder(
        biencoder, dataloader, candidate_encoding, 1, None
    )
    assert labels[0][0] == -1
    assert nns[0][0] == 0, nns[0][0]
    
def test_blink_candidates():
    """
    Tests if the enitity encoding is loaded correctly
    """
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = load_candidates(
        entity_catalogue, 
        entity_encoding, 
        faiss_index=None, 
        index_path=None,
        logger=None,
    )
    assert len(candidate_encoding) == 5903527
    assert len(candidate_encoding[0]) == 1024
    


if __name__ == "__main__":
    test_model_exist()
    test_blink_ner()
    test_blink_biencoder()
    test_blink_candidates()
    test_blink_annotate()