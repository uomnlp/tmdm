import ujson as json

def get_test_docs():
    with open('tests/resources/docs.json', encoding='utf-8') as f:
        return json.load(f)


def get_test_ner():
    with open('tests/resources/ner.json', encoding='utf-8') as f:
        return json.load(f)
