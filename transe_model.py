"""Food.com scoring adapter around PyKEEN's TransE implementation.

This module contains no TransE scoring or embedding implementation. It only
connects PyKEEN's TransE model to this project's user/recipe ID mappings.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pykeen.losses import MarginRankingLoss
from pykeen.models import TransE
from pykeen.triples import TriplesFactory


class TransERecommender(nn.Module):
    """Adapter that scores Food.com triples with PyKEEN's built-in TransE."""

    def __init__(self, triples, entity2id, relation2id, embedding_dim: int,
                 user_entity_ids, recipe_entity_ids, interacted_relation_id: int,
                 distance: str = "l1", margin: float = 1.0, seed: int = 2020):
        super().__init__()
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be 'l1' or 'l2'.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if margin <= 0:
            raise ValueError("margin must be positive.")
        if not 0 <= int(interacted_relation_id) < len(relation2id):
            raise ValueError("INTERACTED relation ID is outside the relation vocabulary.")

        triples_factory = TriplesFactory(
            mapped_triples=np.asarray(triples, dtype=np.int64),
            entity_to_id=entity2id,
            relation_to_id=relation2id,
            num_entities=len(entity2id),
            num_relations=len(relation2id),
        )
        self.kge = TransE(
            triples_factory=triples_factory,
            embedding_dim=int(embedding_dim),
            scoring_fct_norm=1 if distance == "l1" else 2,
            loss=MarginRankingLoss(margin=float(margin)),
            random_seed=int(seed),
        )
        self.distance = distance
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)
        self.interacted_relation_id = int(interacted_relation_id)
        self.register_buffer(
            "user_entity_ids", torch.as_tensor(user_entity_ids, dtype=torch.long))
        self.register_buffer(
            "recipe_entity_ids", torch.as_tensor(recipe_entity_ids, dtype=torch.long))

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        """Delegate plausibility scoring to PyKEEN's TransE model."""
        triples = torch.stack((heads, relations, tails), dim=-1)
        return self.kge.score_hrt(triples).squeeze(-1)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score aligned user/recipe pairs through INTERACTED triples."""
        user_entities = self.user_entity_ids[user_ids]
        recipe_entities = self.recipe_entity_ids[item_ids]
        relations = torch.full_like(user_entities, self.interacted_relation_id)
        return self.score_triples(user_entities, relations, recipe_entities)

    def score_all(self, user_ids: torch.Tensor, item_chunk_size: int = 1024) -> torch.Tensor:
        """Score every recipe for each user, chunking to bound GPU memory."""
        if item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be positive.")
        user_entities = self.user_entity_ids[user_ids]
        relations = torch.full_like(user_entities, self.interacted_relation_id)
        hr_batch = torch.stack((user_entities, relations), dim=-1)
        chunks = [
            self.kge.score_t(hr_batch, tails=self.recipe_entity_ids[start:start + item_chunk_size])
            for start in range(0, len(self.recipe_entity_ids), item_chunk_size)
        ]
        if not chunks:
            return torch.empty((len(user_entities), 0), device=user_entities.device)
        return torch.cat(chunks, dim=1)

    def post_parameter_update(self) -> None:
        """Apply PyKEEN's built-in TransE entity constraint after optimization."""
        self.kge.post_parameter_update()
