import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Intelligence (GoEmotions textcat + aspect rules)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-eval", type=int, default=1000)
    p.add_argument("--labels", type=str, nargs="+", default=["negative", "neutral", "positive"])
    return p.parse_args()


SENT_TO_LABEL = {
    "negative": {"anger", "disgust", "sadness", "disappointment", "remorse", "grief", "annoyance", "disapproval"},
    "positive": {"joy", "admiration", "amusement", "gratitude", "optimism", "love", "approval", "excitement", "surprise", "desire", "relief", "caring", "pride"},
    "neutral": {"neutral", "realization", "curiosity", "confusion", "fear", "nervousness", "embarrassment"},
}


def detect_root():
    input_dir = Path("/kaggle/input")
    for cand in sorted(input_dir.iterdir()):
        if any(cand.rglob("*.csv")):
            return cand
    raise SystemExit(f"no CSV in {input_dir}")


def main():
    from src.data import load_goemotions, aspect_sentiment, top_aspects
    from src.textcat import build_nlp, train_textcat

    args = parse_args()
    base = Path(args.data_root) if args.data_root else detect_root()
    df_train, df_val, label_cols = load_goemotions(base)
    print(f"[DATA] train={len(df_train)} val={len(df_val)} emotion-cols={len(label_cols)}")

    def to_ternary(row):
        active = frozenset(l for l in label_cols if row[l] == 1)
        for label, emos in SENT_TO_LABEL.items():
            if active & emos:
                return label
        return "neutral"

    train_rows = [(t, (to_ternary(r),)) for t, r in zip(df_train["text"], df_train.to_dict("records"))][: args.max_train]
    val_rows = [(t, (to_ternary(r),)) for t, r in zip(df_val["text"], df_val.to_dict("records"))][: args.max_eval]
    label_set = set(SENT_TO_LABEL)
    nlp = build_nlp(label_set)
    textcat_metrics = train_textcat(nlp, train_rows, val_rows, label_set, epochs=args.epochs, log=print)

    aspects = top_aspects(df_val["text"].tolist()[:500])
    per_aspect_sentiment = {}
