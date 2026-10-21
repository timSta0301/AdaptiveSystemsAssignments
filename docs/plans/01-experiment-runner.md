# Person 1 — experiment runner (tasks 1–3)

You own the code that measures `ratio_quality`. Tasks 1, 2, and 3 are the same measurement run four times, so build one runner instead of three scripts. Report prose, the tag method, and the recommender screens are out of scope; those belong to the other two plans.

## Goal

Add a function the others can call:

```python
def evaluate(model: str, section: str) -> dict:
    # model is "tfidf" or "lda"
    # returns ratio_quality, model_seconds, query_seconds
```

Run it for all four cells and leave the numbers where person 2 can copy them:

| model | section       | ratio_quality | model_seconds | query_seconds |
| ----- | ------------- | ------------- | ------------- | ------------- |
| tfidf | Food & Drink  |               |               |               |
| lda   | Food & Drink  |               |               |               |
| tfidf | Sports        |               |               |               |
| lda   | Sports        |               |               |               |

## Data

Load [data/news2.csv](../../data/news2.csv). Use the `description` field for the vectors and `article_section` for the filter. The file says `Food & Drink` (122 articles) and `Sports` (110 articles). The README writes "Food and Drink"; that string matches nothing in the CSV.

## Preprocessing

Match the course examples, do not invent a different pipeline:

- [src/assignment1/examples/tfidf_example.py](../../src/assignment1/examples/tfidf_example.py)
- [src/assignment1/examples/lda_example.py](../../src/assignment1/examples/lda_example.py)

For every description: lowercase, drop NLTK English stopwords, Porter-stem, build a Gensim dictionary and bag-of-words. Build the dictionary on the whole corpus, not only on the target section. Neighbors are searched in the full corpus.

## Models

- `tfidf`: `models.TfidfModel` on the bag-of-words, then the TF-IDF vectors.
- `lda`: `models.LdaModel` with `num_topics=30`, `passes=2`, `random_state=30`, then the topic vectors.

Both paths end in `similarities.MatrixSimilarity(...)`.

## ratio_quality

For every article whose section is the target section:

1. Take its top-10 most similar articles in the corpus.
2. Count how many of those ten are in the same section (`goods`).
3. Add `goods` to `total_goods`.

```text
ratio_quality = total_goods / (num_articles_in_section * 10)
```

Decide explicitly whether an article may be its own nearest neighbor. If the similarity matrix ranks an article against itself first, drop that hit before taking the top 10, and say so next to the results. Otherwise the score is inflated by a perfect self-match.

## Timings

Record two durations, separately, in seconds:

1. Model build: from the start of `evaluate` through the `MatrixSimilarity` call.
2. Query: the pseudocode loop only.

## Suggested place for the code

`src/assignment1/evaluate.py`, plus a small `__main__` that prints the four rows. Keep the examples untouched so the course samples stay as reference.

## Done when

- `evaluate("tfidf", "Food & Drink")` and `evaluate("lda", "Food & Drink")` both return a ratio and both times.
- The same two calls with `"Sports"` return the other two rows.
- The four rows are written down for person 2. A checked-in text or CSV file is enough; the discussion itself is not your job.
