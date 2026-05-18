import unittest

from app.services.menu_embedding import (
    EMBEDDING_MIN_BOOST_SIMILARITY,
    MenuEmbeddingIndex,
    embedding_boost,
)


class MenuEmbeddingIndexTest(unittest.TestCase):
    def test_partial_alias_yields_higher_similarity_than_unrelated_doc(self):
        index = MenuEmbeddingIndex([
            ("fries_01", "감자튀김 감튀 프라이 후렌치후라이 프렌치프라이 side 사이드"),
            ("burger_01", "햄버거 버거 main 식사 든든함 단품"),
        ])
        query_vector = index.query_vector("후렌치 줘")

        fries_sim = index.similarity(query_vector, "fries_01")
        burger_sim = index.similarity(query_vector, "burger_01")

        self.assertGreater(fries_sim, burger_sim)
        self.assertGreaterEqual(fries_sim, EMBEDDING_MIN_BOOST_SIMILARITY)

    def test_unknown_menu_id_returns_zero_similarity(self):
        index = MenuEmbeddingIndex([("fries_01", "감자튀김 감튀")])
        query_vector = index.query_vector("뭐든")

        self.assertEqual(index.similarity(query_vector, "missing_01"), 0.0)

    def test_empty_query_returns_zero_similarity(self):
        index = MenuEmbeddingIndex([("cola_01", "콜라 음료 탄산 차가운 음료")])
        query_vector = index.query_vector("")

        self.assertEqual(index.similarity(query_vector, "cola_01"), 0.0)

    def test_embedding_boost_returns_zero_below_threshold(self):
        self.assertEqual(embedding_boost(0.0), 0)
        self.assertEqual(embedding_boost(EMBEDDING_MIN_BOOST_SIMILARITY - 0.01), 0)

    def test_embedding_boost_scales_above_threshold(self):
        boost_low = embedding_boost(EMBEDDING_MIN_BOOST_SIMILARITY)
        boost_high = embedding_boost(0.8)

        self.assertGreaterEqual(boost_low, 1)
        self.assertGreater(boost_high, boost_low)


if __name__ == "__main__":
    unittest.main()
