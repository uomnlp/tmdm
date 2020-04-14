"""
Setup script adapted from https://github.com/allenai/allennlp-models/blob/master/setup.py
"""
from setuptools import setup, find_packages
import sys

# import os

# version.py defines the VERSION and VERSION_SHORT variables.

VERSION = {}
with open("tmdm/version.py") as version_file:
    exec(version_file.read(), VERSION)

# Load requirements.txt
with open("requirements.txt") as requirements_file:
    lines = requirements_file.read().splitlines()
    install_requirements = []
    for line in lines:
        if '#egg' in line:
            name = line.split("#egg=")[-1]
            install_requirements.append(f"{name} @ {line}")
        else:
            install_requirements.append(line)

# make pytest-runner a conditional requirement,
# per: https://github.com/pytest-dev/pytest-runner#considerations
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup_requirements = [
                         # add other setup requirements as necessary
                     ] + pytest_runner
setup(
    name="tmdm",
    version=VERSION["VERSION"],
    description=("Text mining data model with integration of various annotation format"),
    long_description=open("README.MD").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine reading semantic parsing parsers",
    url="https://github.com/schlevik/tmdm",
    author="Viktor Schlegel",
    author_email="viktor@schlegel-online.de",
    license="GPLv3",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requirements,
    setup_requires=setup_requirements,
    tests_require=["pytest", "flaky", "pytest-cov"],
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False,
)
