# recommender.py

`src/assignment1/recommender.py` is a small content-based news recommender. It reuses `load_articles`, `preprocess` and `parse_tags` from `evaluate.py`.

Each article becomes one TF-IDF vector over its stemmed description words plus its tags. Tags get a `#` prefix (`#food`), so a tag never merges with a description stem of the same spelling. All vectors go into one `MatrixSimilarity` index.

The user profile (`UserProfile`) stores two things:

- `read`: the indices of the articles the user has read.
- `interests`: the average TF-IDF vector of those articles, as `token id -> weight`.

`build_profile(read, vectors)` builds the profile. `recommend(profile, index)` queries the index with the `interests` vector, drops articles that are already in `read`, and returns the 10 most similar ones. `top_interests` lists the highest-weighted tokens of the profile.

## Run

From the repository root, with the packages in `requirements.txt` installed and the NLTK stopword list downloaded:

```bash
python src/assignment1/recommender.py
```

## What it prints

Two example users: one who read the first five `Food & Drink` articles in the CSV and one who read the first five `Sports` articles. For each user:

```text
== user who read 5 Food & Drink articles
read:
  [Food & Drink] <headline>
  ...
interests: <10 highest-weighted stems and #tags>
recommended:
  [<section>] <headline>
  ... (10 lines)
```
