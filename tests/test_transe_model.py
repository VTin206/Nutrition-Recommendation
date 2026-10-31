import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transe_model import TransERecommender, transe_margin_loss


class TransERecommenderTests(unittest.TestCase):
    def make_model(self):
        return TransERecommender(
            n_entities=3, n_relations=1, embedding_dim=2,
            user_entity_ids=[0], recipe_entity_ids=[1, 2],
            interacted_relation_id=0, distance="l1")

    def test_score_is_negative_l1_translation_distance(self):
        model = self.make_model()
        with torch.no_grad():
            model.entity_embedding.weight.copy_(torch.tensor([
                [1.0, -1.0],  # user 0
                [0.0, 0.0],   # recipe 0
                [1.0, 0.0],   # recipe 1
            ]))
            model.relation_embedding.weight.copy_(torch.tensor([[0.5, 0.5]]))
        score = model.score_pairs(torch.tensor([0]), torch.tensor([0]))
        self.assertAlmostEqual(float(score.item()), -2.0)
        self.assertAlmostEqual(float(model.score_pairs(torch.tensor([0]), torch.tensor([1])).item()), -1.0)

    def test_score_all_matches_pair_scoring_and_keeps_item_order(self):
        model = TransERecommender(
            n_entities=7, n_relations=2, embedding_dim=4,
            user_entity_ids=[4, 0], recipe_entity_ids=[5, 2, 6],
            interacted_relation_id=1)
        users = torch.tensor([1, 0])
        actual = model.score_all(users, item_chunk_size=2)
        expected = torch.stack([
            torch.stack([model.score_pairs(users[row:row + 1], torch.tensor([item])).squeeze(0)
                         for item in range(3)])
            for row in range(len(users))
        ])
        torch.testing.assert_close(actual, expected)

    def test_margin_loss_matches_trans_e_hinge_objective(self):
        positive = torch.tensor([-1.0, -0.25], requires_grad=True)
        negative = torch.tensor([-1.5, -2.0], requires_grad=True)
        loss = transe_margin_loss(positive, negative, margin=1.0)
        # [1 - s_positive + s_negative]_+ gives [0.5, 0.0], averaged.
        self.assertAlmostEqual(float(loss.item()), 0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(positive.grad).all())
        self.assertTrue(torch.isfinite(negative.grad).all())

    def test_entity_normalization_enforces_unit_l2_norm(self):
        model = self.make_model()
        with torch.no_grad():
            model.entity_embedding.weight.fill_(3.0)
        model.normalize_entities()
        norms = torch.linalg.vector_norm(model.entity_embedding.weight, ord=2, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms))


if __name__ == "__main__":
    unittest.main()
