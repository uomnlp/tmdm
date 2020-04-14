# tmdm
Text mining data model with integration of various formats, annotation libraries, bells 
and whistles.

## Setup
### PyPi
Execute
```bash
pip install tmdm
```
### From source
```bash
git clone https://github.com/schlevik/tmdm
cd tmdm
pip install -r requirements.txt
pip install . --editable
```

## Example
```python
from tmdm import tmdm_pipeline, add_coref
from tmdm.allennlp.coref import get_coref_provider
nlp = tmdm_pipeline()
add_coref(nlp, provider=get_coref_provider("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"))
doc = nlp("I like cakes. They taste nice.")
assert len(doc._.corefs) == 2
assert doc[2:3]._.get_coref().coreferent(doc[4])
```

Proper documentation is in preparation.