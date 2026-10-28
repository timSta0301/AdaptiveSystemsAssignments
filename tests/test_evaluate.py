import unittest

from evaluate import (
    LDA_PASSES,
    LDA_RANDOM_STATE,
    LDA_TOPICS,
    build_vectors,
    parse_tags,
    preprocess,
    ratio_quality,
    top_neighbors,
)
from gensim import corpora
from gensim.similarities import MatrixSimilarity

# Six food and six sports sentences. Food sentences share "recipe", sports ones share "game".
SMALL_CORPUS = [
    ("pasta recipe with tomato sauce", "Food & Drink"),
    ("fresh bread recipe", "Food & Drink"),
    ("soup recipe for winter", "Food & Drink"),
    ("chocolate cake recipe", "Food & Drink"),
    ("easy salad recipe", "Food & Drink"),
    ("rice recipe with beans", "Food & Drink"),
    ("football game tonight", "Sports"),
    ("tennis game in paris", "Sports"),
    ("the basketball game ended late", "Sports"),
    ("hockey game on ice", "Sports"),
    ("rugby game in the rain", "Sports"),
    ("golf game at the club", "Sports"),
]


def small_corpus_bows():
    texts = [preprocess(text=sentence) for sentence, section in SMALL_CORPUS]
    dictionary = corpora.Dictionary(texts)
    bows = [dictionary.doc2bow(text) for text in texts]
    sections = [section for sentence, section in SMALL_CORPUS]
    return dictionary, bows, sections


class PreprocessTests(unittest.TestCase):
    def test_drops_stopwords_and_stems(self):
        self.assertEqual(preprocess(text="the cooking"), ["cook"])


class TagTests(unittest.TestCase):
    def test_tags_are_trimmed_lowercased_and_empty_ones_dropped(self):
        self.assertEqual(
            parse_tags(text=" celebrity parents,Video,,video "),
            ["celebrity parents", "video", "video"],
        )


class NeighborTests(unittest.TestCase):
    def test_self_match_is_excluded_and_six_of_ten_match_the_section(self):
        # Index 0 is a perfect self-match. The next ten scores are the neighbors.
        sims = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        sections = ["Sports"] + ["Sports"] * 6 + ["Other"] * 4

        neighbors = top_neighbors(sims=sims, query_index=0)

        self.assertEqual(len(neighbors), 10)
        self.assertNotIn(0, neighbors)
        goods = sum(1 for index in neighbors if sections[index] == "Sports")
        self.assertEqual(goods, 6)

    def test_readme_section_name_raises(self):
        # The file stores "Food & Drink". The README wording matches nothing.
        with self.assertRaises(ValueError):
            ratio_quality(
                index=None,
                vectors=[],
                sections=["Food & Drink", "Sports"],
                section="Food and Drink",
            )


class RatioQualityTests(unittest.TestCase):
    def test_ratio_on_a_real_index_leaves_out_the_article_itself(self):
        # A food article has 5 other food articles and 6 sports articles.
        # Only the 5 food ones share a word with it, so its top 10 holds 5 goods.
        # Counting the article itself would give 6 goods and a ratio of 0.6.
        dictionary, bows, sections = small_corpus_bows()
        vectors = build_vectors(model="tfidf", bows=bows, dictionary=dictionary)
        index = MatrixSimilarity(corpus=vectors)

        ratio = ratio_quality(
            index=index, vectors=vectors, sections=sections, section="Food & Drink"
        )

        self.assertAlmostEqual(ratio, 0.5)


class ModelTests(unittest.TestCase):
    def test_lda_settings_match_the_assignment(self):
        self.assertEqual(LDA_TOPICS, 30)
        self.assertEqual(LDA_PASSES, 2)
        self.assertEqual(LDA_RANDOM_STATE, 30)

    def test_tfidf_builds_one_vector_per_sentence(self):
        sentences = [
            "pasta recipe with tomato sauce",
            "the football team won the game",
            "tomato sauce and fresh pasta",
            "game score for the football team",
            "bake the pasta until golden",
            "players train before the game",
        ]
        texts = [preprocess(text=sentence) for sentence in sentences]
        dictionary = corpora.Dictionary(texts)
        bows = [dictionary.doc2bow(text) for text in texts]

        vectors = build_vectors(model="tfidf", bows=bows, dictionary=dictionary)

        self.assertEqual(len(vectors), 6)

    def test_lda_returns_topic_vectors(self):
        dictionary, bows, sections = small_corpus_bows()

        vectors = build_vectors(model="lda", bows=bows, dictionary=dictionary)

        self.assertEqual(len(vectors), len(bows))
        for vector in vectors:
            for topic_id, weight in vector:
                self.assertLess(topic_id, LDA_TOPICS)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            build_vectors(model="word2vec", bows=[], dictionary=None)


if __name__ == "__main__":
    unittest.main()
