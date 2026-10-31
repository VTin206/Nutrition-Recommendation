import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transe_model import TransERecommender


class TransERecommenderTests(unittest.TestCase):
    def test_score_uses_gamma_minus_l1_translation_distance(self):
        model = TransERecommender(n_users=1, n_items=2, embedding_dim=2, gamma=10.0)
        with torch.no_grad():
            model.entity_embedding.weight.copy_(torch.tensor([
                [1.0, -1.0],  # user 0
                [0.0, 0.0],   # recipe 0
                [1.0, 0.0],   # recipe 1
            ]))
            model.relation_embedding.copy_(torch.tensor([[0.5, 0.5]]))
        score = model.score_pairs(torch.tensor([0]), torch.tensor([0]))
        self.assertAlmostEqual(float(score.item()), 8.0)
        self.assertGreater(float(model.score_pairs(torch.tensor([0]), torch.tensor([1])).item()), 8.0)

    def test_score_all_matches_pair_scoring_and_keeps_item_order(self):
        model = TransERecommender(n_users=2, n_items=3, embedding_dim=4)
        users = torch.tensor([1, 0])
        actual = model.score_all(users, item_chunk_size=2)
        expected = torch.stack([
            torch.stack([model.score_pairs(users[row:row + 1], torch.tensor([item])).squeeze(0)
                         for item in range(model.n_items)])
            for row in range(len(users))
        ])
        torch.testing.assert_close(actual, expected)

    def test_pairwise_scores_have_finite_gradients(self):
        model = TransERecommender(n_users=2, n_items=3, embedding_dim=4)
        users = torch.tensor([0, 1])
        safe = model.score_pairs(users, torch.tensor([0, 1]))
        unsafe = model.score_pairs(users, torch.tensor([1, 2]))
        loss = torch.relu(1.0 + unsafe - safe).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.entity_embedding.weight.grad).all())
        self.assertTrue(torch.isfinite(model.relation_embedding.grad).all())


if __name__ == "__main__":
    unittest.main()