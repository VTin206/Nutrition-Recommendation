"""Read and validate the frozen Food.com interaction splits."""
from pathlib import Path

import numpy as np
import torch


def read_split(path, num_items, num_users=None):
    """Read user-to-item rows and reject malformed or duplicate IDs."""
    result = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                user, *items = map(int, line.split())
            except ValueError as error:
                raise ValueError(f"Invalid integer IDs at {path}:{line_number}") from error
            if (user < 0 or (num_users is not None and user >= num_users)
                    or user in result or any(item < 0 or item >= num_items for item in items)
                    or len(set(items)) != len(items)):
                raise ValueError(f"Invalid or duplicate user/item IDs at {path}:{line_number}")
            result[user] = sorted(items)
    if not result:
        raise ValueError(f"No user records found in {path}")
    return result


class Dataset:
    """Train-split catalog and held-out split validation used by the experiment."""

    def __init__(self, train_path, num_items, device):
        split = read_split(train_path, num_items)
        self.n_users, self.m_items = max(split) + 1, int(num_items)
        if set(split) != set(range(self.n_users)) or not all(split.values()):
            raise ValueError("Training split needs contiguous users with nonempty interactions.")
        self.device = torch.device(device)
        self.allPos = [np.asarray(split[user], dtype=np.int64)
                       for user in range(self.n_users)]
        self.trainDataSize = sum(map(len, self.allPos))

    def validate_heldout(self, heldout, other=None):
        for user, items in heldout.items():
            if not 0 <= user < self.n_users:
                raise ValueError("Held-out user is outside the train catalog.")
            if any(item < 0 or item >= self.m_items for item in items):
                raise ValueError("Held-out item is outside the train catalog.")
            if len(set(items)) != len(items):
                raise ValueError("Duplicate held-out user-item pair.")
            if np.intersect1d(items, self.allPos[user]).size:
                raise ValueError("Held-out user-item pairs overlap train.")
            if other is not None and set(items).intersection(other.get(user, [])):
                raise ValueError("Validation and test user-item pairs overlap.")
