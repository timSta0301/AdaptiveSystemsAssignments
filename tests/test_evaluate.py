import unittest

from evaluate import (
    LDA_PASSES,
    LDA_RANDOM_STATE,
    LDA_TOPICS,
    build_vectors,
    preprocess,
    ratio_quality,
    top_neighbors,
)
from gensim import corpora


class PreprocessTests(unittest.TestCase):
    def test_drops_stopwords_and_stems(self):
        self.assertEqual(preprocess(text="the cooking"), ["cook"])


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

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            build_vectors(model="word2vec", bows=[], dictionary=None)


if __name__ == "__main__":
    unittest.main()
