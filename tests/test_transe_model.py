import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from pykeen.losses import MarginRankingLoss
from pykeen.models import TransE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transe_model import TransERecommender


class TransERecommenderTests(unittest.TestCase):
    def make_model(self):
        triples = np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.int64)
        return TransERecommender(
            triples=triples,
            entity2id={"user": 0, "recipe_a": 1, "recipe_b": 2},
            relation2id={"INTERACTED": 0},
            embedding_dim=4,
            user_entity_ids=[0],
            recipe_entity_ids=[1, 2],
            interacted_relation_id=0,
            distance="l1",
            margin=1.0,
            seed=123,
        )

    def test_uses_pykeen_transe_and_margin_ranking_loss(self):
        model = self.make_model()
        self.assertIsInstance(model.kge, TransE)
        self.assertIsInstance(model.kge.loss, MarginRankingLoss)

    def test_triple_scoring_delegates_to_pykeen(self):
        model = self.make_model()
        triples = torch.tensor([[0, 0, 1], [0, 0, 2]], dtype=torch.long)
        actual = model.score_triples(*triples.unbind(dim=1))
        expected = model.kge.score_hrt(triples).squeeze(-1)
        torch.testing.assert_close(actual, expected)

    def test_score_all_matches_pair_scoring_and_preserves_item_order(self):
        model = self.make_model()
        users = torch.tensor([0])
        actual = model.score_all(users, item_chunk_size=1)
        expected = torch.stack([
            model.score_pairs(users, torch.tensor([item])).squeeze(0)
            for item in range(2)
        ]).unsqueeze(0)
        torch.testing.assert_close(actual, expected)

    def test_slcwa_margin_loss_comes_from_pykeen(self):
        model = self.make_model()
        positive = torch.tensor([[-1.0], [-0.25]])
        negative = torch.tensor([[-1.5], [-2.0]])
        loss = model.kge.loss.process_slcwa_scores(positive, negative)
        expected = torch.relu(1.0 - positive + negative).mean()
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
