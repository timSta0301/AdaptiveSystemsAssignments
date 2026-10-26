"""Content-based news recommender (task 5).

Every article is a TF-IDF vector over its description stems and its tags.
The user profile is the reading history plus the average vector of the read
articles. Recommendations are the unread articles most similar to that vector.
"""

from typing import Final, TypedDict

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

from evaluate import (
    DATA_PATH,
    Article,
    Bow,
    SimilarityScores,
    TfidfVector,
    load_articles,
    negative_score,
    parse_tags,
    preprocess,
)

TOP_RECOMMENDATIONS: Final = 10
TOP_INTERESTS: Final = 10
TAG_PREFIX: Final = "#"


class UserProfile(TypedDict):
    read: list[int]
    interests: dict[int, float]


def article_tokens(article: Article) -> list[str]:
    """Description stems plus tags. Tags get a prefix so "#food" never merges with the stem "food"."""
    words: list[str] = preprocess(text=article["description"])
    tags: list[str] = [TAG_PREFIX + tag for tag in parse_tags(text=article["tags"])]
    return words + tags


def build_model(
    articles: list[Article],
) -> tuple[Dictionary, list[TfidfVector], MatrixSimilarity]:
    texts: list[list[str]] = [article_tokens(article=article) for article in articles]
    dictionary: Dictionary = Dictionary(documents=texts)
    bows: list[Bow] = [dictionary.doc2bow(document=text) for text in texts]
    tfidf: TfidfModel = TfidfModel(corpus=bows)
    vectors: list[TfidfVector] = [tfidf[bow] for bow in bows]
    index: MatrixSimilarity = MatrixSimilarity(
        corpus=vectors, num_features=len(dictionary)
    )
    return dictionary, vectors, index


def build_profile(read: list[int], vectors: list[TfidfVector]) -> UserProfile:
    """Average the TF-IDF vectors of every article the user has read."""
    interests: dict[int, float] = {}
    for article_index in read:
        for token_id, weight in vectors[article_index]:
            interests[token_id] = interests.get(token_id, 0.0) + float(weight)
    for token_id in interests:
        interests[token_id] /= len(read)
    return {"read": read, "interests": interests}


def recommend(
    profile: UserProfile, index: MatrixSimilarity, k: int = TOP_RECOMMENDATIONS
) -> list[int]:
    """Unread articles ranked by cosine similarity to the profile vector."""
    query: TfidfVector = list(profile["interests"].items())
    sims: SimilarityScores = index[query]
    ranked = sorted(enumerate(sims), key=negative_score)
    recommendations: list[int] = []
    for article_index, score in ranked:
        if article_index not in profile["read"]:
            recommendations.append(article_index)
        if len(recommendations) == k:
            break
    return recommendations


def top_interests(
    profile: UserProfile, dictionary: Dictionary, k: int = TOP_INTERESTS
) -> list[str]:
    ranked = sorted(profile["interests"].items(), key=negative_score)
    return [dictionary[token_id] for token_id, weight in ranked[:k]]


def show_user(
    name: str,
    read: list[int],
    articles: list[Article],
    dictionary: Dictionary,
    vectors: list[TfidfVector],
    index: MatrixSimilarity,
) -> None:
    profile: UserProfile = build_profile(read=read, vectors=vectors)
    print(f"== {name}")
    print("read:")
    for article_index in read:
        article: Article = articles[article_index]
        print(f"  [{article['article_section']}] {article['headline']}")
    print("interests: " + ", ".join(top_interests(profile=profile, dictionary=dictionary)))
    print("recommended:")
    for article_index in recommend(profile=profile, index=index):
        article = articles[article_index]
        print(f"  [{article['article_section']}] {article['headline']}")
    print()


def main() -> None:
    articles: list[Article] = load_articles(DATA_PATH)
    dictionary, vectors, index = build_model(articles=articles)

    # Two example users: one who read five food articles, one who read five sports articles.
    for section in ("Food & Drink", "Sports"):
        in_section: list[int] = [
            i for i, article in enumerate(articles) if article["article_section"] == section
        ]
        show_user(
            name=f"user who read 5 {section} articles",
            read=in_section[:5],
            articles=articles,
            dictionary=dictionary,
            vectors=vectors,
            index=index,
        )


if __name__ == "__main__":
    main()
