# evaluate.py

`src/assignment1/evaluate.py` measures `ratio_quality` for one model and one news section.

It loads every row of `data/news2.csv`. For `tfidf` and `lda` it stems the `description` field (lowercase, English stopwords removed, Porter stemmer), and builds a Gensim dictionary on the whole corpus. `evaluate("tfidf", section)` turns those bags of words into TF-IDF vectors. `evaluate("lda", section)` uses 30 topics, 2 passes, and `random_state=30`. `evaluate("tags", section)` ignores the description and uses the `tags` field instead: each comma-separated tag is trimmed and lowercased (`parse_tags`), one tag is one token, and the tag lists get TF-IDF weights, so rare tags count more than common ones. All three finish by building `similarities.MatrixSimilarity` on the full corpus, not only on the section.

For each article in the section, the script takes the 10 most similar other articles and counts how many share that section. The article itself is left out of the 10. It is always the nearest neighbor, and keeping it would add one free match to every query.

```text
ratio_quality = total_goods / (num_articles_in_section * 10)
```

The section values in the CSV are `Food & Drink` and `Sports`. `Food and Drink` matches nothing and raises `ValueError`.

## Run

From the repository root, with the packages in `requirements.txt` installed and the NLTK stopword list downloaded:

```bash
python src/assignment1/evaluate.py
```

The script runs six cells: TF-IDF, LDA and tags, each on `Food & Drink` and on `Sports`. Every model is built again for each cell, because the model-build timer starts at the beginning of that call.

## What it prints

One progress line when a cell starts, then one result line when it finishes:

```text
running tfidf / Food & Drink
tfidf,Food & Drink,<ratio_quality>,<model_seconds>,<query_seconds>
```

`model_seconds` is the time from the start of the call through `MatrixSimilarity`. `query_seconds` is only the top-10 loop.

The same six rows are written to `results/ratio_quality_results.csv`. The first line of that file is a comment: self-matches were excluded before the top 10.
