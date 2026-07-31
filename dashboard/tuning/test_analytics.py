import unittest

import numpy as np

from analytics import candidate_table, distance_matrix, summary


def row(trial, score, **params):
    return {"trial": trial, "score": score, "params": params}


class AnalyticsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row(0, 4.0, lr=0.001, mode="a"),
            row(1, 6.0, lr=0.01, mode="a"),
            row(2, 5.5, lr=0.1, mode="b"),
            row(3, 5.8, lr=1.0, mode="a"),
            row(4, 5.7, lr=10.0, mode="b"),
        ]

    def test_distance_is_symmetric_and_bounded(self):
        labels, matrix, ranked = distance_matrix(self.rows, 4)
        self.assertEqual(labels[0], "t001")
        self.assertEqual(len(ranked), 4)
        np.testing.assert_allclose(matrix, matrix.T)
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(np.all(matrix <= 1))

    def test_summary_uses_best_score(self):
        result = summary(self.rows)
        self.assertEqual(result["best"]["trial"], 1)
        self.assertAlmostEqual(result["gap"], 0.2)

    def test_candidate_table_is_ranked(self):
        data, columns = candidate_table(self.rows, 3)
        self.assertEqual(
            [item["trial"] for item in data],
            ["t001", "t003", "t004"],
        )
        self.assertEqual(columns[:3], ["rank", "trial", "score"])


if __name__ == "__main__":
    unittest.main()
