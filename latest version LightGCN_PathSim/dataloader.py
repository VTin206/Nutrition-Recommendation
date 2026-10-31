"""Read frozen Food.com splits and retain all original train graph edges."""
from pathlib import Path

import numpy as np
import torch

DISEASE_KEYS = ('hypertension', 'diabetes', 'heart_failure',
                'steatohepatitis', 'kidney_failure')


def validate_mask(mask, num_items=None):
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.ndim != 1 or not mask.numel()
            or (num_items is not None and len(mask) != num_items)):
        raise ValueError('Expected a Boolean mask aligned to the item catalog.')
    return mask.detach().cpu()


def load_mask(path):
    masks = torch.load(path, weights_only=True, map_location='cpu')
    selected = [validate_mask(masks[name]) for name in DISEASE_KEYS]
    if len({len(value) for value in selected}) != 1:
        raise ValueError('Disease masks have different catalog sizes.')
    mask = torch.stack(selected).all(dim=0)
    if 'comorbidity' in masks and not torch.equal(validate_mask(masks['comorbidity']), mask):
        raise ValueError('Stored all-five mask differs from the five-way intersection.')
    if not mask.any():
        raise ValueError('No nonviolating item remains in the all-five catalog.')
    return mask


def read_split(path, num_items, num_users=None):
    result = {}
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                user, *items = map(int, line.split())
            except ValueError as error:
                raise ValueError(f'Invalid integer IDs at {path}:{line_number}') from error
            if (user < 0 or (num_users is not None and user >= num_users)
                    or user in result or any(i < 0 or i >= num_items for i in items)
                    or len(set(items)) != len(items)):
                raise ValueError(f'Invalid or duplicate user/item IDs at {path}:{line_number}')
            result[user] = sorted(items)
    if not result:
        raise ValueError(f'No user records found in {path}')
    return result


class BasicDataset:
    """Interface name imported by the byte-identical upstream model."""


class Dataset(BasicDataset):
    def __init__(self, train_path, num_items, device):
        split = read_split(train_path, num_items)
        self.n_users, self.m_items = max(split) + 1, int(num_items)
        if set(split) != set(range(self.n_users)) or not all(split.values()):
            raise ValueError('The original catalog requires contiguous users with nonempty train data.')
        self.device = torch.device(device)
        self.allPos = [np.asarray(split[u], dtype=np.int64) for u in range(self.n_users)]
        self.trainDataSize = sum(map(len, self.allPos))
        self._graph = None

    def validate_heldout(self, heldout, other=None):
        for user, items in heldout.items():
            if not 0 <= user < self.n_users:
                raise ValueError('Held-out user is outside the train catalog.')
            if any(i < 0 or i >= self.m_items for i in items):
                raise ValueError('Held-out item is outside the train catalog.')
            if len(set(items)) != len(items):
                raise ValueError('Duplicate held-out user-item pair.')
            if np.intersect1d(items, self.allPos[user]).size:
                raise ValueError('Held-out user-item pairs overlap train.')
            if other is not None and set(items).intersection(other.get(user, [])):
                raise ValueError('Validation and test user-item pairs overlap.')

    def getSparseGraph(self):
        if self._graph is None:
            users = np.repeat(np.arange(self.n_users), [len(p) for p in self.allPos])
            items = np.concatenate(self.allPos) + self.n_users
            edges = torch.tensor(np.stack([np.r_[users, items], np.r_[items, users]]), dtype=torch.long)
            degree = torch.bincount(edges[0], minlength=self.n_users + self.m_items).float()
            inverse = torch.zeros_like(degree)
            inverse[degree > 0] = degree[degree > 0].rsqrt()
            weights = inverse[edges[0]] * inverse[edges[1]]
            self._graph = torch.sparse_coo_tensor(edges, weights,
                (len(degree), len(degree))).coalesce().to(self.device)
        return self._graph
