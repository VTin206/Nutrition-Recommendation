"""TransE embeddings and margin-ranking objective for Food.com KG triples."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TransERecommender(nn.Module):
    """TransE scorer over a multi-relational Food.com knowledge graph.

    Entity and relation IDs are built from the bundled KG. Recommendation scores
    use the ``User --INTERACTED--> Recipe`` relation and are the negative
    translation distance, so larger scores indicate more plausible triples.
    """

    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int,
                 user_entity_ids, recipe_entity_ids, interacted_relation_id: int,
                 distance: str = "l1"):
        super().__init__()
        if min(n_entities, n_relations, embedding_dim) <= 0:
            raise ValueError("Entity/relation counts and embedding_dim must be positive.")
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be 'l1' or 'l2'.")
        if not 0 <= int(interacted_relation_id) < n_relations:
            raise ValueError("INTERACTED relation ID is outside the relation vocabulary.")

        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.embedding_dim = int(embedding_dim)
        self.distance = distance
        self.interacted_relation_id = int(interacted_relation_id)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_dim)
        self.register_buffer("user_entity_ids", torch.as_tensor(user_entity_ids, dtype=torch.long))
        self.register_buffer("recipe_entity_ids", torch.as_tensor(recipe_entity_ids, dtype=torch.long))

        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(self.relation_embedding.weight, -bound, bound)
        with torch.no_grad():
            self.relation_embedding.weight.copy_(
                F.normalize(self.relation_embedding.weight, p=2, dim=1))

    def normalize_entities(self) -> None:
        """Apply TransE's unit-L2 entity constraint before each SGD update."""
        with torch.no_grad():
            self.entity_embedding.weight.copy_(
                F.normalize(self.entity_embedding.weight, p=2, dim=1))

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        """Score aligned triples as negative L1/L2 translation distance."""
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        order = 1 if self.distance == "l1" else 2
        return -torch.linalg.vector_norm(h + r - t, ord=order, dim=-1)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score aligned user/recipe pairs via the INTERACTED relation."""
        users = self.user_entity_ids[user_ids]
        recipes = self.recipe_entity_ids[item_ids]
        relations = torch.full_like(users, self.interacted_relation_id)
        return self.score_triples(users, relations, recipes)

    def score_all(self, user_ids: torch.Tensor, item_chunk_size: int = 1024) -> torch.Tensor:
        """Score each user against the recipe catalog in bounded-size chunks."""
        if item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be positive.")
        user_entities = self.user_entity_ids[user_ids]
        relation = self.relation_embedding.weight[self.interacted_relation_id]
        translated = self.entity_embedding(user_entities) + relation
        chunks = []
        order = 1 if self.distance == "l1" else 2
        for start in range(0, len(self.recipe_entity_ids), item_chunk_size):
            recipe_ids = self.recipe_entity_ids[start:start + item_chunk_size]
            recipes = self.entity_embedding(recipe_ids)
            distances = torch.linalg.vector_norm(
                translated[:, None, :] - recipes[None, :, :], ord=order, dim=-1)
            chunks.append(-distances)
        return torch.cat(chunks, dim=1)


def transe_margin_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor,
                       margin: float) -> torch.Tensor:
    """Mean minibatch form of TransE's pairwise hinge loss.

    Scores are negative distances, so this equals
    ``mean(relu(margin + d_positive - d_negative))``.
    """
    if margin <= 0 or not math.isfinite(margin):
        raise ValueError("margin must be finite and positive.")
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("Positive and negative score tensors must have the same shape.")
    return F.relu(margin - positive_scores + negative_scores).mean()
