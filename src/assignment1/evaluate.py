from csv import DictReader, DictWriter
from pathlib import Path
from time import perf_counter
from typing import Final, Literal, TypeAlias, TypedDict

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
from gensim.similarities import MatrixSimilarity
from nltk import PorterStemmer
from nltk.corpus import stopwords

LDA_TOPICS: Final = 30
LDA_PASSES: Final = 2
LDA_RANDOM_STATE: Final = 30
TOP_NEIGHBORS: Final = 10

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = REPO_ROOT / "data" / "news2.csv"
RESULTS_PATH: Path = REPO_ROOT / "results" / "ratio_quality_results.csv"

STEMMER: PorterStemmer = PorterStemmer()
STOPWORDS: list[str] = stopwords.words("english")

ModelName: TypeAlias = Literal["tfidf", "lda"]
SectionName: TypeAlias = Literal["Food & Drink", "Sports"]
Run: TypeAlias = tuple[ModelName, SectionName]
Bow: TypeAlias = list[tuple[int, int]]
TfidfVector: TypeAlias = list[tuple[int, np.float64]]
LdaVector: TypeAlias = list[tuple[int, np.float32]]
SimilarityScores: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]


class Article(TypedDict):
    description: str
    article_section: str


class Result(TypedDict):
    model: str
    section: str
    ratio_quality: float
    model_seconds: float
    query_seconds: float


RUNS: Final[tuple[Run, Run, Run, Run]] = (
    ("tfidf", "Food & Drink"),
    ("lda", "Food & Drink"),
    ("tfidf", "Sports"),
    ("lda", "Sports"),
)


def load_articles(path: Path) -> list[Article]:
    """Read description and section for every row. Empty descriptions stay in the corpus."""
    articles: list[Article] = []
    with open(file=path, newline="", encoding="utf-8") as handle:
        reader: DictReader[str] = DictReader(handle)
        for row in reader:
            description: str = row["description"] or ""
            article_section: str = row["article_section"] or ""
            article: Article = {
                "description": description,
                "article_section": article_section,
            }
            articles.append(article)
    return articles


def preprocess(text: str) -> list[str]:
    """Lowercase, drop English stopwords, Porter-stem. Same steps as the course examples."""
    return [
        STEMMER.stem(word) for word in text.lower().split() if word not in STOPWORDS
    ]


def build_vectors(
    model: ModelName, bows: list[Bow], dictionary: Dictionary
) -> list[TfidfVector] | list[LdaVector]:
    """Fit TF-IDF or LDA and return one vector per document.

    The vectors are a list so the query loop can read vectors[i] again.
    A transformed corpus can only be iterated once.
    """
    bow: Bow
    if model == "tfidf":
        tfidf: TfidfModel = TfidfModel(corpus=bows)
        vectors: list[TfidfVector] = []
        for bow in bows:
            tfidf_vector: TfidfVector = tfidf[bow]
            vectors.append(tfidf_vector)
        return vectors
    if model == "lda":
        lda: LdaModel = LdaModel(
            corpus=bows,
            num_topics=LDA_TOPICS,
            passes=LDA_PASSES,
            random_state=LDA_RANDOM_STATE,
            id2word=dictionary,
        )
        lda_vectors: list[LdaVector] = []
        for bow in bows:
            lda_vector: LdaVector = lda[bow]
            lda_vectors.append(lda_vector)
        return lda_vectors
    raise ValueError(f"unknown model: {model}")


def negative_score(item: tuple[int, np.float32]) -> np.float32:
    return -item[1]


def top_neighbors(
    sims: SimilarityScores, query_index: int, k: int = TOP_NEIGHBORS
) -> list[int]:
    """Highest-scoring other documents. The article itself is always the nearest hit."""
    ranked: list[tuple[int, np.float32]] = sorted(enumerate(sims), key=negative_score)
    neighbors: list[int] = []
    position: int
    score: np.float32
    for position, score in ranked:
        if position != query_index:
            neighbors.append(position)
    return neighbors[:k]


def ratio_quality(
    index: MatrixSimilarity,
    vectors: list[TfidfVector] | list[LdaVector],
    sections: list[str],
    section: str,
) -> float:
    """Share of top-10 neighbors that belong to the same section.

    ratio_quality = total_goods / (num_articles_in_section * 10)
    """
    targets: list[int] = [i for i, name in enumerate(sections) if name == section]
    if not targets:
        raise ValueError(
            f"no articles found for section {section!r}. "
            "The CSV uses 'Food & Drink', not 'Food and Drink'."
        )

    total_goods: int = 0
    for query_index in targets:
        sims: SimilarityScores = index[vectors[query_index]]
        neighbors: list[int] = top_neighbors(sims=sims, query_index=query_index)
        goods: int = sum(1 for i in neighbors if sections[i] == section)
        total_goods += goods
    return total_goods / (len(targets) * TOP_NEIGHBORS)


def evaluate(model: ModelName, section: str, path: Path = DATA_PATH) -> Result:
    """Build one model on the whole corpus, then score one section.

    model_seconds runs from the start of this call through MatrixSimilarity.
    query_seconds is only the top-10 loop.
    """
    articles: list[Article] = load_articles(path)
    started: float = perf_counter()
    texts: list[list[str]] = [
        preprocess(text=article["description"]) for article in articles
    ]
    dictionary: Dictionary = Dictionary(documents=texts)
    bows: list[Bow] = [dictionary.doc2bow(document=text) for text in texts]
    vectors: list[TfidfVector] | list[LdaVector] = build_vectors(
        model=model, bows=bows, dictionary=dictionary
    )
    similarity_index: MatrixSimilarity = MatrixSimilarity(corpus=vectors)
    model_seconds: float = perf_counter() - started

    started: float = perf_counter()
    sections: list[str] = [article["article_section"] for article in articles]
    ratio: float = ratio_quality(
        index=similarity_index, vectors=vectors, sections=sections, section=section
    )
    query_seconds: float = perf_counter() - started

    return {
        "model": model,
        "section": section,
        "ratio_quality": ratio,
        "model_seconds": model_seconds,
        "query_seconds": query_seconds,
    }


def write_results(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = [
        "model",
        "section",
        "ratio_quality",
        "model_seconds",
        "query_seconds",
    ]
    with open(file=path, mode="w", newline="", encoding="utf-8") as handle:
        handle.write("# self-matches were excluded before the top 10\n")
        writer: DictWriter[str] = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            ratio_text: str = f"{row['ratio_quality']:.6f}"
            model_seconds_text: str = f"{row['model_seconds']:.3f}"
            query_seconds_text: str = f"{row['query_seconds']:.3f}"

            written: dict[str, str] = {
                "model": row["model"],
                "section": row["section"],
                "ratio_quality": ratio_text,
                "model_seconds": model_seconds_text,
                "query_seconds": query_seconds_text,
            }
            writer.writerow(rowdict=written)


def main() -> None:
    rows: list[Result] = []
    model: ModelName
    section: SectionName
    for model, section in RUNS:
        print(f"running {model} / {section}", flush=True)
        row: Result = evaluate(model=model, section=section)
        rows.append(row)
        print(
            f"{row['model']},{row['section']},"
            f"{row['ratio_quality']:.6f},"
            f"{row['model_seconds']:.3f},"
            f"{row['query_seconds']:.3f}",
            flush=True,
        )
    write_results(path=RESULTS_PATH, rows=rows)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
