# Agent instructions

The assignment spec is [README.md](README.md). Read it before changing code.

## What to implement

Tasks 1–3 are code:

- Build TF-IDF vectors from the description fields and compute `ratio_quality` as specified.
- Repeat with LDA. Use 30 topics, 2 passes, and `random_state=30`.
- Run both models for `Food and Drink` and for `Sports`.
- Time two stages separately: model creation, from program start through `similarities.MatrixSimilarity(...)`, and the top-10 similarity loop.

Tasks 4–5 are written answers (tagging method, content-based recommender, low-fidelity screens). Do not draft the report or the prototype unless someone asks.

## Constraints

- Python only. Use Gensim 4.3.3, NLTK 3.9.1, and the standard library, as pinned in [requirements.txt](../../requirements.txt). Do not add scikit-learn, spaCy, or other ML libraries.
- Match the course examples in this folder: plain, human-readable scripts using a Gensim dictionary, TF-IDF or LDA, and `MatrixSimilarity`. Keep new code in this folder. Prefer code a teammate can read and explain line by line.
- When new code in this folder is added or changed, create or update a short note in [docs/](../../docs/) for that code. The note says what the code does, how to run it, and what it prints. Course examples already in this folder do not need notes. Keep tasks 4–5 out of `docs/` unless someone asks.
- The dataset is [data/news2.csv](../../data/news2.csv). Do not commit virtualenvs or lockfiles; those paths are already in [.gitignore](../../.gitignore).
