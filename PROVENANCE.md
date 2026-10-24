# Provenance — Sentiment Intelligence

| Artifact | Source | Proof path |
|----------|--------|------------|
| Kernel v7 (final) | Kaggle `sentiment-intelligence-v7`, GPU, dataset `debarshichanda/goemotions` | `output/run7/output/metrics.json`, kernel log |
| acc 0.513 / macro-F1 0.4948 | same kernel | metrics.json → `textcat` |
| spaCy 3.8 fixes | exclusive_classes → add_pipe default; nlp.initialize sample batch; sgd optimizer | diff run3..run7 logs |
