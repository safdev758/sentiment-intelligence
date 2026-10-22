# Sentiment Intelligence (GoEmotions)

Multilingual-aspect sentiment pipeline: emotion-to-polarity text classifier + rule-based aspect extraction.

## What it does
- **Dataset**: GoEmotions single-label subset → collapsed to ternary polarity (negative / neutral / positive).
- **Model**: spaCy 3.8 `textcat` (bag-of-words, exclusive classes), 10 epochs, compounding batches.
- **Aspect extraction**: lexicon+dependency heuristics (`src/data.py: aspect_sentiment`) for delivery / service / price / quality / product.

## Result (kernel `sentiment-intelligence-v7`, 4k train / 1k val)
- accuracy **0.513**, macro-F1 **0.495** (vs. majority-class baseline ~0.44)
- confusion: negative 229 / neutral 334 / positive 437 (textcat skews positive)
- aspect counts observed: delivery 6, service 4, price 1, quality 1

Full metrics: `output/run7/output/metrics.json`.

## Layout
```
src/textcat.py    pipeline build + train/eval loop
src/data.py       GoEmotions loader, ternary map, aspect/rule extraction
train.py          end-to-end kernel script
tests/            unit tests (mapping, aspect rules)
```

## Kaggle
