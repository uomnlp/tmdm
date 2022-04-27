from typing import Collection
import tempfile

from loguru import logger
from spacy.tokens.doc import Doc
import requests

from tmdm.pipe.pipe import PipeElement
from tmdm.rdf.turtle import to_turtle_all


def upload(docs: Collection[Doc], collection_name, endpoint, debug=None):
    # curl -D -H 'Content-Type: application/x-turtle-RDR' --upload-file rdr_test.ttl -X POST 'http://localhost:9999/blazegraph/sparql'
    payload = to_turtle_all(docs, collection_name)
    if debug is not None:
        with open(debug, 'a+') as f:
            f.write(payload)
    logger.debug(payload)
    res = requests.post(
        url=endpoint,
        data=payload.encode('utf-8'),
        headers={"Content-Type": 'application/x-turtle-RDR'},
    )
    logger.debug(res)


def get_blazegraph_writer(collection_name='default', endpoint='http://localhost:9999/blazegraph/sparql', debug=None):
    return PipeElement(name='graph-writer', field=None, provider=lambda x: upload(x, collection_name=collection_name, endpoint=endpoint, debug=debug))
