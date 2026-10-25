# Spacing review (5-rule framework) — Sentiment Intelligence

## D+2
1. How does the ternary map work (28 emotions → 3 polarities) and what nuance is lost?
2. Why `nlp.initialize(sample)` before update? What breaks without it (nO unset)?
3. What does macro-F1 0.495 vs accuracy 0.513 tell you about class balance?

## D+7
1. Diagnose: why does textcat skew positive on val? Run per-class breakdown locally.
2. How would TF-IDF+SVM compare with a hash-embedding classifier?
3. Where do aspect rules misfire? Test `aspect_sentiment` on 10 crafted inputs.

## D+21 (construct)
1. Train an aspect-BERT head on synth-annotated aspect-sentiment pairs.
2. Pipeline with review ingestion → aspect dashboard; write the two-sentence resume blurb.
3. Multilingual swap: train French/Russian GoEmotions-translation mapped data.

## Pointers
- textcat loop in src/textcat.py; rules src/data.py aspect_sentiment; ternary SENT_TO_LABEL in train.py.
