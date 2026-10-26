# Tasks 4 and 5

## Task 4: comparing articles with tags

The descriptions are free text, so two articles about the same thing can use different words and end up far apart. The tags are short labels that were attached to every article by hand (about 6 per article), so they should be a cleaner signal. To check, we ran the same ratio_quality loop on TF-IDF vectors built from the tag lists instead of the descriptions.

| model | Food & Drink | Sports | model build |
| --- | --- | --- | --- |
| TF-IDF, description | 0.599 | 0.387 | 33-44 s |
| LDA, description | 0.524 | 0.054 | 111-115 s |
| TF-IDF, tags | 0.566 | 0.474 | 0.5 s |

On Sports the tags are clearly the best, and the tag model is built about a hundred times faster because a tag list is only a handful of items. Sports articles share specific tags (sports, nfl, football, rio olympics). On Food & Drink the tags are a bit worse than the description (0.566 against 0.599). Only 65 of the 122 food articles carry the tag "taste", and 11 of them have only tags that no other article has, so there is nothing to match them on. We did not look into it further.

Tags have some problems. The same tag is written in different ways ("Celebrity Moms " and "celebrity moms"), so we strip and lowercase them first. 9,447 of the 12,940 different tags appear on a single article, so they can never help to find a similar one. And related tags such as nfl and football count as different.

To measure the similarity of two articles A and B from their tags we would do this:

1. Take the set of tags of each article (stripped, lowercase, no duplicates).
2. Give every tag a weight idf(t) = log(N / df(t)), where N is the number of articles and df(t) the number of articles that have the tag. A rare tag like "simone biles" (4 articles) gets a high weight, a common one like "entertainment" (183 articles) a low weight.
3. Divide the weight of the tags the two articles share by the weight of all the tags they have between them:

```
sim(A, B) = sum of idf(t) for t in A and B
            -----------------------------
            sum of idf(t) for t in A or B
```

The result goes from 0 (no tag in common) to 1 (same tags). Example with three real articles: X (Aly Raisman note) has the tags aly raisman, laurie hernandez, madison kocian, rio olympics, simone biles. Y (Simone Biles makes a great face) has olympic gymnastics, rio olympics, simone biles, team usa, us womens gymnastics. Z (Simone Biles first pitch) has houston astros, olympics, simone biles. X and Y share two tags and have 8 different tags in total, which gives 0.213. X and Z share only "simone biles" out of 7 in total, which gives 0.134. So Y is closer to X than Z, as expected.

In the code (`evaluate.py`, model "tags") we did not implement this formula. We build TF-IDF vectors from the tags and compare them with cosine similarity like for the descriptions, which also gives more weight to rare tags. The numbers in the table are from that version. The two ideas are close but the exact scores would differ.

Tags and description can also be used together. In `recommender.py` each tag is added to the words of the article with a "#" in front, so one vector contains both.

## Task 5: content-based recommender for news

We process every article like in the course (lowercase, remove stopwords, Porter stemmer) and add its tags as extra words. Gensim turns the words into a dictionary and TF-IDF vectors, and `MatrixSimilarity` gives the similarity of any vector to all articles. This is what `recommender.py` does.

The user model is what we know about one reader. At the moment we store the list of articles the user read and the average of their TF-IDF vectors, which is the "interest" vector. To recommend, we compare that vector with all articles, remove the ones already read and show the 10 most similar. We can also show why: the words or tags with the highest weight in the interest vector, e.g. "because you read about bagels and avocado".

What we would store additionally (not implemented):

- the topics or tags the user picks when starting, so a new user does not get an empty page
- likes and dislikes (a dislike lowers the score of similar articles)
- how long the user stayed on an article, so a long read counts more than a click
- the date of each read, so older reads count less
- tags or sections the user chose to hide

In `results/recommendation.txt` we tried two users who each read five articles of one section. The food reader got 7 Food & Drink articles out of 10, the others were about bagels (Weird News), dinner conversation (Comedy) and the Whole30 diet (Wellness). The sports reader got 6 Sports articles out of 10, plus for example a DWTS article (Entertainment) and a politics article. The recommender finds the right area but not always the right section, because it only compares words and the average of five articles is a bit blurry. Hiding or disliking would help here.

Low-fidelity screens:

```
1. Choose interests            2. For you (home)
+--------------------------+   +--------------------------+
| What do you like to read?|   | For you | Latest     (Me) |
|                          |   |--------------------------|
| [Sports] [Food & Drink]  |   | [img] Healthy salad      |
| [Politics] [Travel]      |   | Food & Drink #recipes    |
| [Wellness] [Business]    |   | because you read: rice   |
| [Comedy]  [Parenting]    |   | [like] [not for me]      |
|                          |   |--------------------------|
| [Skip]        [Continue] |   | [img] Biles withdraws    |
+--------------------------+   | Sports #olympics         |
                               | because you read: Raisman|
                               | [like] [not for me]      |
                               +--------------------------+

3. Article                     4. My interests
+--------------------------+   +--------------------------+
| < back            [save] |   | < My interests           |
| Biles withdraws from ... |   |                          |
| Sports                   |   | sports      ########  [x]|
| text text text text ...  |   | nba         #####     [x]|
| text text text text ...  |   | olympics    ####      [x]|
| Tags: #olympics          |   | [+ add]                  |
|       #simone biles      |   |                          |
|--------------------------|   | Hidden: celebrity gossip |
| More like this           |   | History (12)   [view]    |
| > Biles makes a face     |   | [reset profile]          |
| > Raisman note           |   +--------------------------+
| [like]     [not for me]  |
+--------------------------+
```

The user starts on screen 1, then uses the home feed (2). Opening a card shows the article (3), and reading, liking or disliking it changes the next feed. "Me" opens screen 4, where the user can see the interests we have for them and remove or hide them.
