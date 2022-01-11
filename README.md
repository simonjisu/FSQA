# A Framework for Enhancing Boardroom TODS with Financial Ontology and Embedded Machine Learning

A demonstration about how does the framework might working in a broad perspective.

![demo](./figs/demo.gif)

## prerequisite

```plain
postgresql >= 14.1
python >= 3.8
```

You can install package by:

```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm
```

## How to Run

```
$ cd ./src/ && streamlit run app.py
```