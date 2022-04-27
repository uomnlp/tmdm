import re
from typing import Collection

from spacy import displacy
from spacy.tokens import Doc

from tmdm import NamedEntity
from tmdm.pipe.pipe import PipeElement


def html_writer(docs: Collection[Doc], file='stdout', prefix=None):
    docs = list(docs)
    output = [{
        "text": d.text,
        'ents': [{"start": ne.start_char, "end": ne.end_char, "label": ne.label_,
                  # "kb_id": ne._.ne_meta['uri'].split('/')[-1], "kb_url": ne._.ne_meta['uri']
                  } for ne in d._.nes],
        'title': d._.id
    } for d in docs]
    out = displacy.render(output, style='ent', manual=True)
    ### HACK for spacy 2
    ents_iter = iter(e for d in docs for e in d._.nes)

    def subber(match):
        next_ent: NamedEntity = next(ents_iter)
        kb_url = next_ent._.ne_meta.get('uri')
        if kb_url:
            kb_id = kb_url.split('/')[-1]
            if prefix:
                kb_url = f"{prefix}/{kb_id}"
            return f'<a style="text-decoration: none; color: inherit; font-weight: normal" href="{kb_url}">{kb_id}</a></mark>'
        else:
            return '</mark>'
    out = re.sub(r'</mark>', subber, out)

    if file == 'stdout':
        print(out)
    else:
        with open(file, 'a+') as f:
            f.write(out)

    return docs

def get_html_writer_pipe(out_file=None, prefix=None):
    return PipeElement(name='html-writer', field=None, provider=lambda x: html_writer(x, file=out_file, prefix=prefix))
