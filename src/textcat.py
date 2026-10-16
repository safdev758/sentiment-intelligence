from collections import defaultdict

import random

import spacy
from spacy.util import minibatch, compounding


def build_nlp(labels):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    for l in sorted(labels):
        textcat.add_label(l)
