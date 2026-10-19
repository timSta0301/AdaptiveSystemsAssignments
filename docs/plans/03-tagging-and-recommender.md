# Person 3 — tagging and the recommender (tasks 4 and 5)

You own task 4 and task 5. Neither one waits on the TF-IDF or LDA runner. Do not build a working user interface: task 5 asks for a low-fidelity prototype, which means wireframes in the report.

Task 5 is in scope because this team is Erasmus students of the HCID master.

## Task 4 — similarity from tags

Every row in [data/news2.csv](../../data/news2.csv) already has a `tags` field. Tags are comma-separated, for example `united states,Women,Retire Well,United States Travel,National Parks,vacation`.

Propose this method in the report:

1. Split `tags` on commas.
2. Normalize each tag: trim whitespace and lowercase, so `Women` and `women` match.
3. Treat the result as a set. Duplicate tags on one article count once.
4. Score two articles with Jaccard similarity: `|A ∩ B| / |A ∪ B|`. If both sets are empty, define the score as 0.
5. Say how this sits next to the description similarity from tasks 1–3. Tags recover shared entities and categories that the body vectors miss. Description vectors still cover topical wording that was never tagged. A simple combination is a weighted sum of the two scores, with the weights stated explicitly.

You do not have to ship a full second experiment. A short worked example on two real articles from the CSV is enough to show the score is computable. If you do add code, keep it separate from person 1's runner, for example `src/assignment1/tag_similarity.py`.

## Task 5 — content-based news recommender

Explain a content-based recommender that reuses the techniques already in the assignment. No collaborative filtering, and no claim that you trained it on this CSV.

### User profile

State which information you would store, and why each field is there:

- Identifiers of articles the user has read or saved.
- Accumulated section counts, so a user who mostly reads Sports is not treated like a cold start.
- Accumulated normalized tags, as a multiset or as weights, using the same normalization as task 4.
- Optional: a TF-IDF or LDA vector averaged from the descriptions of articles the user engaged with. If you keep an LDA vector, it has to be in the same 30-topic space as the corpus model (`num_topics=30`, `passes=2`, `random_state=30`).

Also say what you deliberately leave out. A content-based profile does not need other users' ratings.

### Ranking

For an unseen article, score it against the profile and sort descending:

- Tag Jaccard (or cosine on the tag weights) between the article and the profile's tag weights.
- Cosine between the article's TF-IDF or LDA vector and the profile vector, when you store one.
- Combine them with the same weighted sum you described in task 4, and suppress articles the user has already read.

### Low-fidelity prototype

Draw three screens. Paper photos or a Figma frame are both fine. Put the images in the report. A coded UI is not required.

- **Feed.** A ranked list of recommended articles: headline, section, and the tags that matched the profile. A control to refresh or to explain "why this".
- **Article.** Headline, section, tags, and the description, plus an action that updates the profile (read or save).
- **Preferences.** The stored profile made visible: top sections, top tags, and a way to remove a tag or a section the user does not want.

Each screen should show sample content drawn from the CSV, not lorem ipsum, so it is obvious the prototype is this news set.

## Done when

- The report states the tag-similarity formula, the normalization rules, and how it combines with description similarity.
- The report describes the profile fields and the ranking steps.
- The report includes the three screens, and none of them is a running application.
