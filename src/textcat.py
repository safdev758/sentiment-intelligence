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

    rng = random.Random(42)
    optimizer = nlp.create_optimizer()
    sample = [spacy.training.Example.from_dict(nlp.make_doc(t), {"cats": to_cats(g, label_set)}) for t, g in train_rows[:32]]
    nlp.initialize(lambda: sample)
    losses = {}
    for epoch in range(1, epochs + 1):
        shuffled = list(train_rows)
        rng.shuffle(shuffled)
        for batch in minibatch(shuffled, size=compounding(4.0, 64.0, 1.001)):
            examples_per_batch = []
            for text, g in batch:
                doc = nlp.make_doc(text)
                exs = spacy.training.Example.from_dict(doc, {"cats": to_cats(g, label_set)})
                examples_per_batch.append(exs)
            nlp.update(examples_per_batch, sgd=optimizer, losses=losses, drop=0.3)
        log(f"[SEN] epoch {epoch}/{epochs} loss={losses.get('textcat', 0):.4f}")
    y_true, y_pred = [], []
    for _, t in val_rows:
        y_true.append(t[0])
    preds = []
    for doc in nlp.pipe([t for t, _ in val_rows]):
        best = max(doc.cats.items(), key=lambda kv: kv[1])
        preds.append(best[0])
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    log(f"[SEN-EVAL] acc={acc:.4f} macro-f1={f1:.4f}")
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "confusion": dict(sorted((l, preds.count(l)) for l in label_set))}
