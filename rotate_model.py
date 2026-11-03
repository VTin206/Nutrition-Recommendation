"""Food.com adapter around PyKEEN's RotatE implementation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pykeen.losses import NSSALoss
from pykeen.models import RotatE
from pykeen.triples import TriplesFactory


class RotatERecommender(nn.Module):
    """Adapt PyKEEN RotatE to score Food.com user/recipe interactions."""

    def __init__(self, triples, entity2id, relation2id, embedding_dim: int,
                 gamma: float, adversarial_temperature: float,
                 user_entity_ids, recipe_entity_ids, interacted_relation_id: int,
                 seed: int = 2020):
        super().__init__()
        if embedding_dim <= 0 or gamma <= 0 or adversarial_temperature <= 0:
            raise ValueError("embedding_dim, gamma, and adversarial_temperature must be positive.")
        if not 0 <= int(interacted_relation_id) < len(relation2id):
            raise ValueError("INTERACTED relation ID is outside the relation vocabulary.")

        triples_factory = TriplesFactory(
            mapped_triples=np.asarray(triples, dtype=np.int64),
            entity_to_id=entity2id,
            relation_to_id=relation2id,
            num_entities=len(entity2id),
            num_relations=len(relation2id),
        )
        self.kge = RotatE(
            triples_factory=triples_factory,
            embedding_dim=int(embedding_dim),
            loss=NSSALoss(
                margin=float(gamma),
                adversarial_temperature=float(adversarial_temperature),
            ),
            random_seed=int(seed),
        )
        self.embedding_dim = int(embedding_dim)
        self.gamma = float(gamma)
        self.adversarial_temperature = float(adversarial_temperature)
        self.interacted_relation_id = int(interacted_relation_id)
        self.register_buffer(
            "user_entity_ids", torch.as_tensor(user_entity_ids, dtype=torch.long))
        self.register_buffer(
            "recipe_entity_ids", torch.as_tensor(recipe_entity_ids, dtype=torch.long))

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        """Delegate triple scoring to PyKEEN RotatE."""
        triples = torch.stack((heads, relations, tails), dim=-1)
        return self.kge.score_hrt(triples).squeeze(-1)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score aligned user/recipe pairs through INTERACTED triples."""
        users = self.user_entity_ids[user_ids]
        recipes = self.recipe_entity_ids[item_ids]
        relations = torch.full_like(users, self.interacted_relation_id)
        return self.score_triples(users, relations, recipes)

    def score_all(self, user_ids: torch.Tensor, item_chunk_size: int = 1024) -> torch.Tensor:
        """Score each user against every recipe, chunking to bound memory."""
        if item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be positive.")
        users = self.user_entity_ids[user_ids]
        relations = torch.full_like(users, self.interacted_relation_id)
        hr_batch = torch.stack((users, relations), dim=-1)
        chunks = [
            self.kge.score_t(
                hr_batch, tails=self.recipe_entity_ids[start:start + item_chunk_size])
            for start in range(0, len(self.recipe_entity_ids), item_chunk_size)
        ]
        if not chunks:
            return torch.empty((len(users), 0), device=users.device)
        return torch.cat(chunks, dim=1)

    def post_parameter_update(self) -> None:
        """Apply PyKEEN's relation constraint after an optimizer update."""
        self.kge.post_parameter_update()
