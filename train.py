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
