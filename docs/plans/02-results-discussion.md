# Person 2 — results discussion (tasks 1–3)

You own the written answers for tasks 1, 2, and 3. Person 1 owns the runner that produces the four numbers. You can draft the argument before those numbers exist and drop them in at the end. The tag method and the recommender screens are not yours.

## What you turn in

A section of the team report with:

1. The TF-IDF `ratio_quality` for Food & Drink, and the two execution times (model build, then the top-10 loop).
2. The same three figures for LDA on Food & Drink, then why the score and the times differ from TF-IDF.
3. Both models on Sports, then why that quality is worse than Food & Drink.

Copy the figures from person 1. Do not reimplement the models unless their runner is missing a cell.

## Results table

Leave this empty until the runner finishes, then fill every cell. Section strings must match the CSV: `Food & Drink`, not "Food and Drink".

| model | section      | ratio_quality | model_seconds | query_seconds |
| ----- | ------------ | ------------- | ------------- | ------------- |
| tfidf | Food & Drink |               |               |               |
| lda   | Food & Drink |               |               |               |
| tfidf | Sports       |               |               |               |
| lda   | Sports       |               |               |               |

`model_seconds` is everything up to and including `similarities.MatrixSimilarity(...)`. `query_seconds` is only the pseudocode loop. Say that in the report so the two columns are not compared as if they were the same kind of work.

## Task 2 — why TF-IDF and LDA differ

Draft this before the numbers arrive, then check it against the table.

- TF-IDF keeps a dimension per stemmed word. Two articles are close when they share distinctive words. Food and drink writing has a lot of those (ingredients, recipes, restaurants), so neighbors often stay inside the section.
- LDA compresses every article into 30 topics (`num_topics=30`, `passes=2`, `random_state=30`). Words that TF-IDF treats as different can land in the same topic, so a neighbor can match on a broad topic and fall outside the section. The ratio can move in either direction; report the direction you actually measure and tie the explanation to it.
- The model-build time should be higher for LDA, because training the topic model is the extra work. The query time is then a comparison of 30-dimensional topic vectors against a large sparse TF-IDF vocabulary. Say which one was faster in your run rather than asserting it in advance.

## Task 3 — why Sports is worse

Food & Drink has 122 articles and Sports has 110, so the gap is not "Sports is a smaller class." The denominator of `ratio_quality` already normalizes by the number of articles in the section.

Sports language overlaps politics, entertainment, and general news (names, games, "win", "team", cities). A Sports article's nearest neighbors are more often outside the section. Food and drink vocabulary is more specific to that section, so a larger share of the top 10 stays inside it. Use one or two real headlines from [data/news2.csv](../../data/news2.csv) as illustrations if they make that concrete.

Compare both models. If LDA narrows or widens the gap versus TF-IDF, say so from the table instead of from this prediction.

## Done when

- The report states all four ratios and all eight times.
- The TF-IDF versus LDA paragraph explains both the quality difference and the timing difference you measured.
- The Sports paragraph explains the worse quality without blaming the section size.
