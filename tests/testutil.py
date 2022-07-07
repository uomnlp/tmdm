import ujson as json

def get_test_docs():
    with open('tests/resources/docs.json', encoding='utf-8') as f:
        return json.load(f)


def get_test_ner():
    with open('tests/resources/ner.json', encoding='utf-8') as f:
        return json.load(f)
    
def get_test_brat():
    brat_txt = ""
    brat_ann = ""
    with open('tests/resources/test_brat/test_brat_doc.txt', encoding='utf-8') as f:
        brat_txt = f.read()
    with open('tests/resources/test_brat/test_brat_ann.ann', encoding='utf-8') as f:
        brat_ann = f.read()
        
    return [brat_txt, brat_ann]
