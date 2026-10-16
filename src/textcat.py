from collections import defaultdict

import random

import spacy
from spacy.util import minibatch, compounding


def build_nlp(labels):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    for l in sorted(labels):
        textcat.add_label(l)
    return nlp


def to_cats(row, labels):
    pos = row if isinstance(row, str) else (row[0] if isinstance(row, (list, tuple)) and row else None)
    labels_sorted = sorted(labels)
    cats = {l: l == pos for l in labels_sorted}
    return cats


def train_textcat(nlp, train_rows, val_rows, label_set, epochs=10, log=print):
    from sklearn.metrics import accuracy_score, f1_score

