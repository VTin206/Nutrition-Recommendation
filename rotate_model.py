"""Food.com adapter for the official RotatE implementation.

The scoring model and filtered negative-sampling dataset are loaded from the
vendored, pinned KnowledgeGraphEmbedding source. This module only adapts the
Food.com user/item ID spaces to the upstream triple interface.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
UPSTREAM_CODES = HERE / "vendor" / "KnowledgeGraphEmbedding" / "codes"
UPSTREAM_COMMIT = "2e440e0f9c687314d5ff67ead68ce985dc446e3a"


def _load_upstream():
    """Load original modules without shadowing this project's dataloader."""
    previous_dataloader = sys.modules.pop("dataloader", None)
    try:
        loader_spec = importlib.util.spec_from_file_location(
            "foodcom_rotate_upstream_dataloader", UPSTREAM_CODES / "dataloader.py")
        loader = importlib.util.module_from_spec(loader_spec)
        sys.modules[loader_spec.name] = loader
        loader_spec.loader.exec_module(loader)
        # The upstream model imports `TestDataset` from a top-level dataloader.
        sys.modules["dataloader"] = loader

        model_spec = importlib.util.spec_from_file_location(
            "foodcom_rotate_upstream_model", UPSTREAM_CODES / "model.py")
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = model_module
        model_spec.loader.exec_module(model_module)
    finally:
        sys.modules.pop("dataloader", None)
        if previous_dataloader is not None:
            sys.modules["dataloader"] = previous_dataloader
    return model_module.KGEModel, loader.TrainDataset, loader.BidirectionalOneShotIterator


KGEModel, TrainDataset, BidirectionalOneShotIterator = _load_upstream()


class RotatERecommender(nn.Module):
    """Delegate all RotatE scoring to the original KGEModel implementation."""

    def __init__(self, nentity: int, nrelation: int, hidden_dim: int, gamma: float,
                 user_entity_ids, recipe_entity_ids, interacted_relation_id: int):
        super().__init__()
        if hidden_dim <= 0 or gamma <= 0:
            raise ValueError("RotatE hidden_dim and gamma must be positive.")
        if not 0 <= int(interacted_relation_id) < nrelation:
            raise ValueError("INTERACTED relation ID is outside the relation vocabulary.")

        self.kge = KGEModel(
            model_name="RotatE",
            nentity=int(nentity),
            nrelation=int(nrelation),
            hidden_dim=int(hidden_dim),
            gamma=float(gamma),
            double_entity_embedding=True,
            double_relation_embedding=False,
        )
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.interacted_relation_id = int(interacted_relation_id)
        self.register_buffer("user_entity_ids", torch.as_tensor(user_entity_ids, dtype=torch.long))
        self.register_buffer("recipe_entity_ids", torch.as_tensor(recipe_entity_ids, dtype=torch.long))

    def forward(self, sample, mode="single"):
        return self.kge(sample, mode=mode)

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        triples = torch.stack((heads, relations, tails), dim=-1)
        return self.kge(triples).squeeze(-1)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        users = self.user_entity_ids[user_ids]
        recipes = self.recipe_entity_ids[item_ids]
        relations = torch.full_like(users, self.interacted_relation_id)
        return self.score_triples(users, relations, recipes)

    def score_all(self, user_ids: torch.Tensor, item_chunk_size: int = 1024) -> torch.Tensor:
        """Score each user against every recipe via upstream tail-batch scoring."""
        if item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be positive.")
        users = self.user_entity_ids[user_ids]
        relations = torch.full_like(users, self.interacted_relation_id)
        positive = torch.stack((users, relations, torch.zeros_like(users)), dim=1)
        chunks = []
        for start in range(0, len(self.recipe_entity_ids), item_chunk_size):
            tails = self.recipe_entity_ids[start:start + item_chunk_size]
            candidates = tails.unsqueeze(0).expand(len(users), -1).contiguous()
            chunks.append(self.kge((positive, candidates), mode="tail-batch"))
        if not chunks:
            return torch.empty((len(users), 0), device=users.device)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def post_parameter_update() -> None:
        """The upstream RotatE implementation applies no post-step projection."""
        return None
