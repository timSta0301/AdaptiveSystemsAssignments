import unittest

from recommender import build_model, build_profile, recommend


def article(description, tags, section):
    return {
        "headline": description,
        "tags": tags,
        "description": description,
        "article_section": section,
    }


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.articles = [
            article("pasta recipe with tomato sauce", "food,pasta", "Food & Drink"),
            article("fresh pasta and tomato salad", "food,pasta", "Food & Drink"),
            article("the football team won the game", "sports,football", "Sports"),
            article(
                "players train before the football game", "sports,football", "Sports"
            ),
        ]
        self.dictionary, self.vectors, self.index = build_model(articles=self.articles)

    def test_profile_averages_the_read_vectors(self):
        profile = build_profile(read=[0, 0], vectors=self.vectors)
        expected = {token_id: float(weight) for token_id, weight in self.vectors[0]}
        for token_id, weight in expected.items():
            self.assertAlmostEqual(profile["interests"][token_id], weight, places=6)

    def test_read_articles_are_never_recommended(self):
        profile = build_profile(read=[0], vectors=self.vectors)
        recommendations = recommend(profile=profile, index=self.index, k=3)
        self.assertNotIn(0, recommendations)
        self.assertEqual(recommendations[0], 1)

    def test_order_of_the_interest_vector_does_not_matter(self):
        # The profile is built in reading order, not sorted by token id.
        profile = build_profile(read=[0, 2], vectors=self.vectors)
        reversed_profile = {
            "read": profile["read"],
            "interests": dict(reversed(list(profile["interests"].items()))),
        }

        self.assertEqual(
            recommend(profile=profile, index=self.index, k=2),
            recommend(profile=reversed_profile, index=self.index, k=2),
        )


if __name__ == "__main__":
    unittest.main()
