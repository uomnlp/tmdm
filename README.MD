# Sample project
This project shows how to properly layout your project, write test cases, write your `setup.py`, generate documentation and install the package.

## Setup
Execute
```bash
pip install -r requirements.txt
```
for development (i.e. testing, doc generation etc), execute
```bash
pip install -r requirements-dev.txt
```

## Tests
Execute
```bash
python setup.py test
```
from the project root dir. This will run all tests in `./tests` folder and generate a code coverage report.

Adapt appropriate options (i.e. packages you want to have a code coverage report for) in `setup.cfg`.

## Doc
Run
```bash
sphinx-apidoc sample -o docs
```
to generate the documentation of your API and put it into the `./docs` folder. (This was already done for this project).

To generate the html version, run
```bash
python setup.py build_sphinx
```

Adapt appropriate entries (project name, author, version)in ``/docs/conf.py``

## Installation
To install the project into your python environment, run
```bash
pip install .
```
Adapt `setup.py` (project name, author, license, etc) to your needs.
