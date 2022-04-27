from typing import List, Collection

from loguru import logger
from spacy.tokens import Doc, Span

header_tpl = """@prefix : <http://tanc.manchester.ac.uk/{coll}/> .
@prefix tanc: <http://tanc.manchester.ac.uk/> .
@prefix dbr: <http://dbpedia.org/resource/> .
@prefix DBpedia: <http://dbpedia.org/ontology/> .
@prefix Schema: <http://schema.org/> .
@prefix Wikidata: <https://www.wikidata.org/wiki/> .
@prefix DUL: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .
"""


def to_turtle(doc: Doc) -> List[str]:
    result = []
    result.append(f':{doc._.id} tanc:text "{str(doc)}"^^xsd:string .')
    for ne in doc._.nes:
        ne: Span
        # TODO: for now linking only identified mentions
        if ne._.ne_meta:
            s, e = ne.start_char, ne.end_char
            result.append(f"<<:{doc._.id} tanc:mentions <{ne._.ne_meta['uri']}>>> tanc:similarity {ne._.ne_meta['similarity']};")
            for t in ne._.ne_meta['types']:
                if any(t.startswith(x) for x in ['DBpedia', 'Schema', 'Wikidata', 'DUL', 'Http']):
                    if t.lower().startswith('http'):
                        t = f'<{t.lower()}>'
                    result.append(f"\t\t rdfs:type {t} ;")
                else:
                    logger.debug(f"Unknown type prefix: {t}")
            result.append(f"\t\t tanc:start {s} ;")
            result.append(f"\t\t tanc:end {e} ;")
            result.append(f"\t\t tanc:support {str(ne._.ne_meta['support'])} .")
    return result


def to_turtle_all(docs: Collection[Doc], collection_name):
    x = [header_tpl.format(coll=collection_name)]
    x.extend(r for doc in docs for r in to_turtle(doc))
    return '\n'.join(x)
