import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="keyphrasetransformer",
    version="0.0.2",
    license="apache-2.0",
    author="Shivanand Roy",
    author_email="snrcodes@gmail.com",
    description="Quickly extract key-phrases/topics from you text data with T5 transformer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shivanandroy/KeyPhraseTransformer",
    project_urls={
        "Repo": "https://github.com/Shivanandroy/KeyPhraseTransformer",
        "Bug Tracker": "https://github.com/Shivanandroy/KeyPhraseTransformer/issues",
    },
    keywords=[
        "keyword extraction",
        "keyphrase extraction",
        "keyphrase",
        "extraction"
        "T5",
        "simpleT5",
        "transformers", 
        "NLP"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "nltk",
        "transformers"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)