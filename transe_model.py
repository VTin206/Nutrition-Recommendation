"""TransE scoring backbone for safety-aware user-recipe recommendation."""
from __future__ import annotations

import math

import torch
from torch import nn


class TransERecommender(nn.Module):
    """One-relation TransE scorer over user--INTERACTED-->recipe triples.

    Scores follow the official TransE convention: gamma minus the L1 translation
    distance. User entities occupy [0, n_users); recipe entities follow them.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int,
                 gamma: float = 12.0):
        super().__init__()
        if min(n_users, n_items, embedding_dim) <= 0:
            raise ValueError("Entity counts and embedding_dim must be positive.")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.gamma = float(gamma)
        self.entity_embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_dim)
        self.relation_embedding = nn.Parameter(torch.empty(1, self.embedding_dim))
        embedding_range = (self.gamma + 2.0) / self.embedding_dim
        nn.init.uniform_(self.entity_embedding.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.relation_embedding, -embedding_range, embedding_range)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return scores for aligned user/item IDs; larger means more preferred."""
        user = self.entity_embedding(user_ids)
        item_entities = item_ids + self.n_users
        item = self.entity_embedding(item_entities)
        translated = user + self.relation_embedding
        return self.gamma - torch.abs(translated - item).sum(dim=-1)

    def score_all(self, user_ids: torch.Tensor, item_chunk_size: int = 1024) -> torch.Tensor:
        """Score a batch of users against the catalog without a large 3-D allocation."""
        if item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be positive.")
        translated = self.entity_embedding(user_ids) + self.relation_embedding
        item_ids = torch.arange(self.n_items, device=user_ids.device)
        item_chunks = []
        for start in range(0, self.n_items, item_chunk_size):
            item_entities = item_ids[start:start + item_chunk_size] + self.n_users
            items = self.entity_embedding(item_entities)
            distance = torch.abs(translated[:, None, :] - items[None, :, :]).sum(dim=-1)
            item_chunks.append(self.gamma - distance)
        return torch.cat(item_chunks, dim=1)