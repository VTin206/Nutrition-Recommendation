"""Sparse PathSim baselines for Food.com.

PathSim follows Sun et al., "PathSim: Meta Path-Based Top-K Similarity Search
in Heterogeneous Information Networks," PVLDB 4(11), 2011:
https://www.vldb.org/pvldb/vol4/p992-sun.pdf

This is a local implementation of the paper's symmetric meta-path formula,
not a port of a third-party package.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


PATHSIM_CONFIG = {
    "reference": "Sun et al., PVLDB 4(11), 2011",
    "reference_url": "https://www.vldb.org/pvldb/vol4/p992-sun.pdf",
    "formula": "2 * count_P(x,y) / (count_P(x,x) + count_P(y,y))",
    "meta_paths": {
        "pathsim_user": "User -INTERACTED-> Recipe -INTERACTED^-1-> User",
        "pathsim_ingredient": "Recipe -HAS_INGREDIENT-> Ingredient -HAS_INGREDIENT^-1-> Recipe",
    },
    "interaction_edges": "Frozen training split only; binary positive interactions",
    "ingredient_edges": "Binary HAS_INGREDIENT triples from the bundled KG",
    "path_count_matrix": "A @ A.T for each binary source-target adjacency matrix",
    "neighbor_pruning": "None; all nonzero PathSim similarities are retained",
    "recommendation_score": "PathSim-weighted sum of neighboring users' or recipes' train interactions",
    "self_similarity": "Diagonal removed before recommendation scoring",
    "nutrient_meta_path": "Excluded: current KG stores seven nutrient type links per recipe, not nutrient values",
    "candidate_policy": "Shared evaluator candidates exclude each user's train and validation items",
    "inference_variants": "Unfiltered ranking and SAFE-only hard filter; UNKNOWN is excluded by the filter",
    "tie_break": "Descending score, then ascending item ID",
}


def _pathsim(adjacency: csr_matrix) -> csr_matrix:
    """Compute exact sparse PathSim for a binary symmetric meta-path."""
    adjacency = adjacency.tocsr().astype(np.float32)
    adjacency.sum_duplicates()
    if adjacency.nnz:
        adjacency.data[:] = 1.0

    counts = (adjacency @ adjacency.T).tocsr()
    visibility = np.asarray(counts.diagonal(), dtype=np.float32)
    counts.setdiag(0)
    counts.eliminate_zeros()

    pairs = counts.tocoo(copy=False)
    denominator = visibility[pairs.row] + visibility[pairs.col]
    valid = denominator > 0
    values = np.zeros_like(pairs.data, dtype=np.float32)
    values[valid] = 2.0 * pairs.data[valid] / denominator[valid]
    return csr_matrix((values, (pairs.row, pairs.col)), shape=counts.shape)


def build_pathsim_scores(dataset, kg):
    """Build user-user and recipe-ingredient PathSim ranking score matrices."""
    user_rows = np.repeat(np.arange(dataset.n_users, dtype=np.int64),
                          [len(items) for items in dataset.allPos])
    train_items = np.concatenate(dataset.allPos).astype(np.int64, copy=False)
    user_item = csr_matrix(
        (np.ones(len(train_items), dtype=np.float32), (user_rows, train_items)),
        shape=(dataset.n_users, dataset.m_items), dtype=np.float32)
    user_item.sum_duplicates()
    if user_item.nnz:
        user_item.data[:] = 1.0

    user_similarity = _pathsim(user_item)
    user_scores = (user_similarity @ user_item).toarray().astype(np.float32, copy=False)

    triples = kg["triples"]
    ingredient_triples = triples[
        triples[:, 1] == kg["has_ingredient_relation_id"]]
    if not len(ingredient_triples):
        raise ValueError("KG has no HAS_INGREDIENT triples for PathSim.")

    recipe_entity_to_item = np.full(len(kg["entity2id"]), -1, dtype=np.int64)
    recipe_entity_to_item[kg["recipe_entity_ids"]] = np.arange(dataset.m_items)
    recipe_rows = recipe_entity_to_item[ingredient_triples[:, 0]]
    if np.any(recipe_rows < 0):
        raise ValueError("HAS_INGREDIENT triples must have recipe heads.")
    ingredient_entities, ingredient_cols = np.unique(
        ingredient_triples[:, 2], return_inverse=True)
    recipe_ingredient = csr_matrix(
        (np.ones(len(recipe_rows), dtype=np.float32),
         (recipe_rows, ingredient_cols)),
        shape=(dataset.m_items, len(ingredient_entities)), dtype=np.float32)
    recipe_ingredient.sum_duplicates()
    if recipe_ingredient.nnz:
        recipe_ingredient.data[:] = 1.0

    recipe_similarity = _pathsim(recipe_ingredient)
    ingredient_scores = (user_item @ recipe_similarity).toarray().astype(
        np.float32, copy=False)
    return ({"pathsim_user": user_scores,
             "pathsim_ingredient": ingredient_scores},
            {"users": dataset.n_users,
             "recipes": dataset.m_items,
             "train_interaction_edges": int(user_item.nnz),
             "ingredient_nodes": int(len(ingredient_entities)),
             "ingredient_edges": int(recipe_ingredient.nnz)})
