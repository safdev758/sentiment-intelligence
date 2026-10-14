import json
import random
from pathlib import Path

import pandas as pd

ASPECTS = {
    "delivery": ("deliver", "shipping", "shipped", "courier", "package delay", "late", "tracking"),
    "price": ("price", "expensive", "cheap", "cost", "worth"),
    "quality": ("quality", "broke", "broke apart", "durable", "sturdy", "defect"),
    "service": ("support", "customer service", "refund", "complaint", "return"),
}
ASPECT_ORDER = list(ASPECTS)


def load_goemotions(root: Path, valid_frac=0.15, seed=42):
    """GoEmotions CSVs: columns text,label (label index into emotion taxonomy)."""
    parts = []
    for p in sorted(root.rglob("*.csv"))[:3]:
        parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    label_cols = [c for c in df.columns if c not in ("text", "id", "author", "example_very_unclear")]
    df = df[~df["example_very_unclear"].astype(bool)] if "example_very_unclear" in df.columns else df
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    cut = int(len(df) * valid_frac)
    return df.iloc[idx[cut:]], df.iloc[idx[:cut]], label_cols


def aspect_sentiment(text: str):
    """Rule-based aspect extraction from complaint sentences: (aspect, matches)."""
