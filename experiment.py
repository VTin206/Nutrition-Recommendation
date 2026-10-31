"""M1/M2 Food.com experiments using TransE margin training."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safety_aware import (SafetyStatus, StatusLookup, assess_safety, dcg,
                          experiment_arms, parse_profile_conditions, prepare_metric_reference, safe_ndcg)

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = HERE
sys.path.insert(0, str(ROOT_DEFAULT / "latest version LightGCN_PathSim"))
from dataloader import Dataset, read_split
from transe_model import TransERecommender, transe_margin_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_kg(path, n_users, n_items, train, val, test):
    """Load the bundled training KG and verify its interaction split boundary."""
    frame = pd.read_csv(path, sep="\t", usecols=["head", "relation", "tail"], dtype=str)
    if frame.empty or frame.isna().any().any():
        raise ValueError("KG triples must be non-empty and have no missing values.")
    required_users = [f"User_{u}" for u in range(n_users)]
    required_recipes = [f"Recipe_{i}" for i in range(n_items)]
    entities = sorted(set(frame["head"]) | set(frame["tail"]) |
                      set(required_users) | set(required_recipes))
    relations = sorted(set(frame["relation"]))
    entity2id = {name: idx for idx, name in enumerate(entities)}
    relation2id = {name: idx for idx, name in enumerate(relations)}
    if "INTERACTED" not in relation2id:
        raise ValueError("KG is missing the INTERACTED relation.")
    triples = np.column_stack((frame["head"].map(entity2id),
                               frame["relation"].map(relation2id),
                               frame["tail"].map(entity2id))).astype(np.int64)
    expected_train = {(f"User_{u}", f"Recipe_{i}") for u, items in train.items() for i in items}
    heldout = {(f"User_{u}", f"Recipe_{i}") for split in (val, test)
               for u, items in split.items() for i in items}
    interaction_rows = frame.loc[frame["relation"] == "INTERACTED"]
    actual_interactions = set(zip(interaction_rows["head"], interaction_rows["tail"]))
    missing = expected_train - actual_interactions
    extra = actual_interactions - expected_train
    leaked = heldout & actual_interactions
    if missing:
        raise ValueError(f"KG is missing {len(missing)} frozen training interactions.")
    if leaked:
        raise ValueError(f"KG contains {len(leaked)} validation/test interactions.")
    if extra:
        raise ValueError(f"KG contains {len(extra)} INTERACTED triples outside the frozen train split.")
    return {
        "triples": triples,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "user_entity_ids": np.asarray([entity2id[x] for x in required_users], dtype=np.int64),
        "recipe_entity_ids": np.asarray([entity2id[x] for x in required_recipes], dtype=np.int64),
        "interacted_relation_id": relation2id["INTERACTED"],
    }


class KnownTripleSampler:
    """Uniformly corrupt one TransE triple while rejecting known KG facts."""

    def __init__(self, triples, n_entities, n_relations, seed):
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.rng = np.random.default_rng(seed)
        self.known_codes = np.unique(self._encode(triples[:, 0], triples[:, 1], triples[:, 2]))

    def _encode(self, heads, relations, tails):
        return (heads * self.n_relations + relations) * self.n_entities + tails

    def sample(self, batch):
        if not len(batch):
            raise ValueError("Cannot sample negatives for an empty triple batch.")
        negatives = batch.copy()
        corrupt_head = self.rng.random(len(batch)) < 0.5
        candidates = self.rng.integers(0, self.n_entities, size=len(batch), dtype=np.int64)
        negatives[corrupt_head, 0] = candidates[corrupt_head]
        negatives[~corrupt_head, 2] = candidates[~corrupt_head]
        for _ in range(1000):
            codes = self._encode(negatives[:, 0], negatives[:, 1], negatives[:, 2])
            positions = np.searchsorted(self.known_codes, codes)
            safe_positions = np.minimum(positions, len(self.known_codes) - 1)
            invalid = ((positions < len(self.known_codes)) &
                       (self.known_codes[safe_positions] == codes))
            if not invalid.any():
                return negatives
            rows = np.flatnonzero(invalid)
            replacements = self.rng.integers(0, self.n_entities, size=len(rows), dtype=np.int64)
            negatives[rows[corrupt_head[rows]], 0] = replacements[corrupt_head[rows]]
            negatives[rows[~corrupt_head[rows]], 2] = replacements[~corrupt_head[rows]]
        raise RuntimeError("Could not sample corrupted TransE triples.")


def load_inputs(root):
    root = Path(root).resolve()
    names = ["interactions/foodcom_train.csv", "interactions/foodcom_test.csv",
             "interactions/foodcom_train.txt", "interactions/foodcom_val.txt",
             "interactions/foodcom_test.txt", "recipes/foodcom_recipes_metadata.csv",
             "health/clinical_health_rules.json", "kg/foodcom_kg_triples.tsv"]
    paths = [root / name for name in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    metadata = pd.read_csv(paths[5])
    nutrients = ["calories", "fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv"]
    if "item_id" not in metadata or any(c not in metadata for c in nutrients):
        raise ValueError("Metadata needs item_id and calories/fat/sugar/sodium/protein fields.")
    metadata.item_id = metadata.item_id.astype(int)
    if metadata.item_id.duplicated().any():
        raise ValueError("Recipe item_id values must be unique.")
    metadata = metadata.sort_values("item_id").reset_index(drop=True)
    n_items = int(metadata.item_id.max()) + 1
    if set(metadata.item_id) != set(range(n_items)):
        raise ValueError("item_id must be contiguous from zero.")
    rules = json.loads(paths[6].read_text(encoding="utf-8-sig"))
    train_frame, test_frame = pd.read_csv(paths[0]), pd.read_csv(paths[1])
    for label, frame in (("train", train_frame), ("test", test_frame)):
        if not {"user_id", "item_id", "rating"}.issubset(frame.columns):
            raise ValueError(f"{label} CSV must include user_id,item_id,rating.")
        frame.user_id = frame.user_id.astype(int)
        frame.item_id = frame.item_id.astype(int)
        if frame.duplicated(["user_id", "item_id"]).any():
            raise ValueError(f"{label} CSV contains duplicate user-item interactions.")
        frame.rating = pd.to_numeric(frame.rating, errors="raise")
        if not frame.rating.isin([4.0, 5.0]).all():
            raise ValueError(f"{label} split contains ratings outside the frozen 4/5-star preference labels.")
    n_users = max(int(train_frame.user_id.max()), int(test_frame.user_id.max())) + 1
    train, val, test = (read_split(p, n_items, n_users) for p in paths[2:5])
    pairs = lambda frame: {(int(u), int(i)) for u, i in zip(frame.user_id, frame.item_id)}
    split_pairs = lambda split: {(u, i) for u, items in split.items() for i in items}
    if pairs(train_frame) != split_pairs(train) or pairs(test_frame) != split_pairs(test):
        raise ValueError("CSV interactions differ from frozen train/test text splits.")
    for first, second, label in ((train, val, "train/validation"),
                                 (train, test, "train/test"), (val, test, "validation/test")):
        if any(set(items).intersection(second.get(u, [])) for u, items in first.items()):
            raise ValueError(f"Leakage found between {label} splits.")
    kg = load_kg(paths[7], n_users, n_items, train, val, test)
    return (root, metadata, rules, train_frame, test_frame, train, val, test, paths, kg)


def make_health_scenario(health_condition, rules, n_users):
    """Apply one selected health-condition scenario to every user in this run."""
    conditions = parse_profile_conditions(health_condition)
    if not conditions:
        raise ValueError("Choose a disease for this run, e.g. hypertension or diabetes; 'none' is not a disease scenario.")
    missing = [condition for condition in conditions if condition not in rules.get("diseases", {})]
    if missing:
        raise ValueError("No configured health rule for: " + ", ".join(missing))
    return conditions, {user: conditions for user in range(n_users)}


def build_safety_index(metadata, rules, user_conditions, n_users):
    """Build item safety labels for the selected run-wide health scenario."""
    columns = ["calories", "fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv"]
    nutrition = metadata.set_index("item_id")[columns].to_dict("records")
    vectors, user_keys = {}, {}
    for user in range(n_users):
        conditions = user_conditions.get(user)
        key = None if conditions is None else tuple(conditions)
        user_keys[user] = key
        if key not in vectors:
            vectors[key] = np.asarray(
                [assess_safety(conditions, row, rules).value for row in nutrition], dtype=object)
    return user_keys, vectors


def status_vector(user_keys, vectors, user):
    key = user_keys.get(user)
    if key in vectors:
        return vectors[key]
    return np.full(len(next(iter(vectors.values()))), SafetyStatus.UNKNOWN.value, dtype=object)


def training_safety_pools(dataset, user_keys, vectors, raw_train):
    """Build safety pairs only from train positives and unobserved recipes."""
    safe_positive, unsafe, safety_users = {}, {}, []
    catalog = np.arange(dataset.m_items, dtype=np.int64)
    for user, observed in enumerate(dataset.allPos):
        states = status_vector(user_keys, vectors, user)
        allowed = np.ones(dataset.m_items, dtype=bool)
        train_seen = raw_train.get(user, [])
        if len(train_seen):
            allowed[train_seen] = False
        safe_items = observed[states[observed] == SafetyStatus.SAFE.value]
        unsafe_items = catalog[(states == SafetyStatus.UNSAFE.value) & allowed]
        if len(safe_items) and len(unsafe_items):
            safe_positive[user] = safe_items
            unsafe[user] = unsafe_items
            safety_users.append(user)
    return safe_positive, unsafe, np.asarray(safety_users, dtype=np.int64)


def build_model(kg, embedding, distance="l1"):
    return TransERecommender(
        len(kg["entity2id"]), len(kg["relation2id"]), embedding,
        kg["user_entity_ids"], kg["recipe_entity_ids"],
        kg["interacted_relation_id"], distance=distance)


def pair_scores(model, user_ids, item_ids):
    return model.score_pairs(user_ids, item_ids)


def score_user_batch(model, user_ids, item_chunk_size=1024):
    return model.score_all(user_ids, item_chunk_size=item_chunk_size)


def validation_score(model, dataset, val, raw_train, user_keys, vectors, k=10):
    """Binary safe-conditioned validation NDCG under the run-wide disease scenario."""
    active, relevant_by_user = [], {}
    for user, heldout in val.items():
        states = status_vector(user_keys, vectors, user)
        relevant = [item for item in heldout if states[item] == SafetyStatus.SAFE.value]
        if relevant:
            active.append(user)
            relevant_by_user[user] = set(relevant)
    if not active:
        raise ValueError("Validation has no SAFE positive under this health scenario; cannot select checkpoint.")
    model.eval()
    values_by_user = []
    with torch.no_grad():
        for start in range(0, len(active), 64):
            user_batch = active[start:start + 64]
            ids = torch.as_tensor(user_batch, dtype=torch.long, device=dataset.device)
            score_batch = score_user_batch(model, ids).cpu().numpy()
            for row, user in enumerate(user_batch):
                states = status_vector(user_keys, vectors, user)
                seen = np.zeros(dataset.m_items, dtype=bool)
                seen[raw_train.get(user, [])] = True
                candidates = np.flatnonzero((states == SafetyStatus.SAFE.value) & ~seen)
                if not len(candidates):
                    continue
                scores = score_batch[row, candidates]
                count = min(k, len(candidates))
                top = np.argpartition(-scores, count - 1)[:count]
                order = np.lexsort((candidates[top], -scores[top]))
                ranked = candidates[top[order]]
                relevant = relevant_by_user[user]
                dcg = sum(1.0 / math.log2(rank + 2)
                          for rank, item in enumerate(ranked) if item in relevant)
                idcg = sum(1.0 / math.log2(rank + 2)
                           for rank in range(min(k, len(relevant))))
                if idcg:
                    values_by_user.append(dcg / idcg)
    if not values_by_user:
        raise ValueError("No eligible validation candidates remain for checkpoint selection.")
    return float(np.mean(values_by_user))


def input_hashes(root, paths):
    sources = [root / "latest version LightGCN_PathSim/dataloader.py",
               HERE / "transe_model.py", HERE / "experiment.py", HERE / "safety_aware.py"]
    return {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest(p)
            for p in paths + sources}

def train(args):
    (root, metadata, rules, _, _, raw_train, val, _, paths, kg) = load_inputs(args.root)
    n_users = max(raw_train) + 1
    conditions, user_conditions = make_health_scenario(args.health_condition, rules, n_users)
    user_keys, vectors = build_safety_index(metadata, rules, user_conditions, n_users)
    checkpoint_selector = "validation safe-conditioned NDCG@10"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = Dataset(root / "interactions/foodcom_train.txt", len(metadata), device)
    dataset.validate_heldout(val)
    safe_positive, unsafe, safety_users = training_safety_pools(
        dataset, user_keys, vectors, raw_train)
    if args.model == "m2" and not len(safety_users):
        raise ValueError("M2 needs safe training interactions and unsafe recipe candidates under this health scenario.")
    seed_all(args.seed)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"architecture": "TransE", "training_objective": "margin_ranking",
                "health_condition": conditions,
                "distance": args.distance, "margin": args.margin,
                "embedding": args.embedding, "entity_count": len(kg["entity2id"]),
                "relation_count": len(kg["relation2id"]),
                "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "checkpoint_selector": checkpoint_selector,
                "input_sha256": input_hashes(root, paths)}
    manifest_path = run_dir / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Run folder inputs/settings differ. Choose a fresh run folder.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out = run_dir / args.model
    out.mkdir(exist_ok=False)
    model_config = {"model": args.model, "architecture": "TransE",
                    "training_objective": "TransE margin ranking loss",
                    "health_condition": conditions,
                    "distance": args.distance, "margin": args.margin,
                    "lambda_safe": args.lambda_safe if args.model == "m2" else 0.,
                    "delta": args.delta if args.model == "m2" else 0.}
    (out / "training_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    model = build_model(kg, args.embedding, args.distance).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    safety_rng = np.random.default_rng(args.seed + 1)
    triple_sampler = KnownTripleSampler(
        kg["triples"], len(kg["entity2id"]), len(kg["relation2id"]), args.seed)
    history, best, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(kg["triples"]))
        sums, count, safety_count = {"transe_loss": 0., "safety_loss": 0., "total_loss": 0.}, 0, 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = kg["triples"][indices]
            negative_batch = triple_sampler.sample(batch)
            as_tensor = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
            positive_tensor, negative_tensor = as_tensor(batch), as_tensor(negative_batch)
            model.normalize_entities()
            positive_scores = model.score_triples(*positive_tensor.unbind(dim=1))
            negative_scores = model.score_triples(*negative_tensor.unbind(dim=1))
            kg_loss = transe_margin_loss(positive_scores, negative_scores, args.margin)
            safe_loss = torch.zeros((), device=device)
            safety_pairs = 0
            if args.model == "m2":
                safety_batch_users = safety_rng.choice(
                    safety_users, size=len(batch), replace=True)
                safe_items = np.asarray([
                    safety_rng.choice(safe_positive[int(user)]) for user in safety_batch_users
                ], dtype=np.int64)
                unsafe_items = np.asarray([
                    safety_rng.choice(unsafe[int(user)]) for user in safety_batch_users
                ], dtype=np.int64)
                user_tensor = as_tensor(safety_batch_users)
                safe_scores = pair_scores(model, user_tensor, as_tensor(safe_items))
                unsafe_scores = pair_scores(model, user_tensor, as_tensor(unsafe_items))
                safe_loss = torch.relu(args.delta + unsafe_scores - safe_scores).mean()
                safety_pairs = len(batch)
            total = kg_loss + args.lambda_safe * safe_loss if args.model == "m2" else kg_loss
            if not torch.isfinite(total):
                raise ValueError("Non-finite loss; stopping training.")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            n = len(batch)
            sums["transe_loss"] += float(kg_loss.detach()) * n
            sums["safety_loss"] += float(safe_loss.detach()) * safety_pairs
            safety_count += safety_pairs
            sums["total_loss"] += float(total.detach()) * n
            count += n
        model.normalize_entities()
        row = {"epoch": epoch, "transe_loss": sums["transe_loss"] / count,
               "safety_loss": sums["safety_loss"] / safety_count if safety_count else 0.0,
               "total_loss": sums["total_loss"] / count}
        row["validation_safe_ndcg_at_10"] = validation_score(
            model, dataset, val, raw_train, user_keys, vectors)
        selection_score = row["validation_safe_ndcg_at_10"]
        history.append(row)
        if selection_score > best:
            best, best_epoch, stale = selection_score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation_score": best, "manifest": manifest, "model_config": model_config},
                       out / "best_checkpoint.pt")
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    plot_loss(history, out / "loss_curves.png")
    (out / "training_summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_validation_score": best,
        "checkpoint_selection_metric": manifest["checkpoint_selector"],
        "best_validation_safe_ndcg_at_10": best,
        "epochs_completed": len(history), "health_condition": conditions,
        "scenario_users": n_users,
        "kg_triples": int(len(kg["triples"])),
        "training_users_with_safety_pairs": int(len(safety_users)), "device": str(device),
        "architecture": "TransE", "training_objective": "margin_ranking"}, indent=2),
        encoding="utf-8")
    print(f"Saved {args.model.upper()} checkpoint at epoch {best_epoch}: {out}")


def plot_loss(history, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; saved CSV loss history only.", flush=True)
        return
    data = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("transe_loss", "TransE margin"), ("safety_loss", "Safety"),
                       ("total_loss", "Total")]:
        ax.plot(data.epoch, data[col], label=label)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training loss components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def score_model(model, dataset):
    model.eval()
    chunks = []
    for start in range(0, dataset.n_users, 64):
        user_ids = torch.arange(start, min(start + 64, dataset.n_users), device=dataset.device)
        chunks.append(score_user_batch(model, user_ids, item_chunk_size=512).cpu())
    return torch.cat(chunks).numpy()

def evaluate(args):
    (root, metadata, rules, _, test_frame, raw_train, val, test, paths, kg) = load_inputs(args.root)
    n_users = max(raw_train) + 1
    conditions, user_conditions = make_health_scenario(args.health_condition, rules, n_users)
    user_keys, vectors = build_safety_index(metadata, rules, user_conditions, n_users)
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "experiment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hashes(root, paths):
        raise ValueError("Evaluation data/code hashes differ from training.")
    if manifest.get("health_condition") != conditions:
        raise ValueError("Evaluation health condition differs from training. Use the same --health-condition.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = Dataset(root / "interactions/foodcom_train.txt", len(metadata), device)
    dataset.validate_heldout(test)
    arms = experiment_arms()
    available_models = {
        model_name for _, model_name, _ in arms
        if (run_dir / model_name / "best_checkpoint.pt").is_file()
    }
    if not available_models:
        raise FileNotFoundError(f"No trained TransE checkpoint found under {run_dir}.")
    arms = [arm for arm in arms if arm[1] in available_models]
    relevance = {}
    for row in test_frame.itertuples():
        if float(row.rating) >= 4.0:
            relevance.setdefault(int(row.user_id), {})[int(row.item_id)] = float(row.rating) - 3.0
    contexts = {}
    for user in range(n_users):
        states = status_vector(user_keys, vectors, user)
        excluded = set(raw_train.get(user, [])) | set(val.get(user, []))
        available = np.ones(dataset.m_items, dtype=bool)
        if excluded:
            available[list(excluded)] = False
        candidates = np.flatnonzero(available)
        lookup = StatusLookup(states)
        reference = prepare_metric_reference(relevance.get(user, {}), lookup, candidates)
        contexts[user] = (states, candidates, lookup, reference)
    example_users = set(range(min(n_users, 200)))
    examples, rows, score_cache = [], [], {}
    for label, model_name, hard_filter in arms:
        if model_name not in score_cache:
            checkpoint = torch.load(run_dir / model_name / "best_checkpoint.pt",
                                    map_location=device, weights_only=False)
            model = build_model(kg, manifest["embedding"], manifest["distance"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            score_cache[model_name] = score_model(model, dataset)
        scores = score_cache[model_name]
        for user in range(n_users):
            states, candidates, user_status, reference = contexts[user]
            order = np.lexsort((candidates, -scores[user, candidates]))
            unfiltered = candidates[order].tolist()
            ranked = [i for i in unfiltered if states[i] == SafetyStatus.SAFE.value] if hard_filter else unfiltered
            user_status = StatusLookup(states)
            if user in example_users:
                if label == "M1":
                    for rank, item in enumerate(unfiltered[:20], 1):
                        if states[item] != SafetyStatus.SAFE.value:
                            examples.append({"user_id": user, "kind": "baseline_non_safe_top20",
                                "item_id": item, "item_name": metadata.iloc[item].get("name", ""),
                                "safety_status": states[item], "score": float(scores[user, item]),
                                "baseline_rank": rank})
            if hard_filter and any(states[i] != SafetyStatus.SAFE.value for i in ranked[:20]):
                raise AssertionError("Hard filter emitted a non-SAFE item.")
            for k in (5, 10, 20):
                recs = ranked[:k]
                base = safe_ndcg(recs, relevance.get(user, {}), user_status, candidates, k, 5, reference)
                rec_rel = [float(relevance.get(i, 0.0)) for i in recs]
                safe_flags = [user_status.get(i, SafetyStatus.UNKNOWN) == SafetyStatus.SAFE for i in recs]
                for penalty in (1, 3, 5, 10):
                    metrics = dict(base)
                    metrics["P-DCG@K"] = dcg(rec_rel, k, safe_flags, penalty)
                    metrics["P-NDCG@K"] = (metrics["P-DCG@K"] / base["P_IDCG@K"]
                        if base["P_IDCG@K"] > 0 else None)
                    rows.append({"health_condition": "|".join(conditions),
                        "system": label, "model": model_name, "hard_filter": hard_filter,
                        "user_id": user, "K": k, "penalty_P": penalty, "catalog_size": dataset.m_items,
                        "coverage_status": ("FULL_TOP_K" if len(recs) == k else
                            ("INSUFFICIENT_SAFE_CANDIDATES" if hard_filter else "INSUFFICIENT_CANDIDATES")),
                        "recommended": recs,
                        "safe_returned_count": sum(user_status.get(i) == SafetyStatus.SAFE for i in recs),
                        **metrics})
        print(f"Evaluated {label}", flush=True)
    pd.DataFrame(rows).to_csv(run_dir / "per_user_metrics.csv", index=False)
    table = aggregate_rows(rows)
    table.to_csv(run_dir / "model_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(run_dir / "recommendation_examples.csv", index=False)
    comparison_examples(rows, run_dir)
    plot_metrics(table, run_dir)
    print(table.to_string(index=False))

def aggregate_rows(rows):
    frame = pd.DataFrame(rows)
    metrics = ["Safety@K", "SafetyAmongReturned", "FillRate@K", "P-DCG@K", "P-NDCG@K",
        "NDCG@K", "Precision@K", "Recall@K", "EligiblePrecision@K", "EligibleRecall@K",
        "Unsafe@1", "Unknown@K", "RecommendationCount"]
    group_cols = ["health_condition", "system", "model", "hard_filter", "K", "penalty_P"]
    output = []
    for key, part in frame.groupby(group_cols, dropna=False, sort=False):
        record = dict(zip(group_cols, key))
        record["users_evaluated"] = int(part.user_id.nunique())
        record["full_list_users"] = int(part["Safety@K"].notna().sum())
        record["short_list_users"] = int(part.coverage_status.str.startswith("INSUFFICIENT").sum())
        record["undefined_P_NDCG_users"] = int(part["P-NDCG@K"].isna().sum())
        for metric in metrics:
            record[metric] = part[metric].mean() if part[metric].notna().any() else np.nan
        n_recs = int(part.RecommendationCount.sum())
        record["PooledSafetyAmongReturned"] = int(part.safe_returned_count.sum()) / n_recs if n_recs else np.nan
        distinct = {item for recs in part.recommended for item in recs}
        record["RecommendationCoverage"] = len(distinct) / int(part.catalog_size.iloc[0])
        output.append(record)
    return pd.DataFrame(output)


def plot_metrics(table, folder):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; saved comparison tables only.", flush=True)
        return
    charts = [("Safety@K", "safety_at_k.png"), ("P-NDCG@K", "p_ndcg_at_k.png"),
        ("NDCG@K", "ndcg_at_k.png"), ("Recall@K", "recall_at_k.png"),
        ("FillRate@K", "fill_rate_at_k.png")]
    for metric, filename in charts:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, group in table[table.penalty_P == 5].groupby("system"):
            ax.plot(group.K, group[metric], marker="o", label=name)
        ax.set(xlabel="K", ylabel=metric, title=f"{metric} by system (P=5)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)


def comparison_examples(rows, folder):
    frame = pd.DataFrame(rows)
    frame = frame[(frame.K == 10) & (frame.penalty_P == 5)]
    frame.pivot(index="user_id", columns="system",
        values=["recommended", "Safety@K", "NDCG@K", "P-NDCG@K"]).to_csv(
            folder / "user_level_examples.csv")


def sweep(args):
    """Sweep lambda/delta and choose only by validation safe-conditioned NDCG."""
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for lam in (0.1, 0.5, 1.0, 2.0, 5.0):
        for delta in (0.5, 1.0, 2.0):
            run = out / f"lambda_{lam:g}_delta_{delta:g}"
            one = argparse.Namespace(**vars(args))
            one.model, one.lambda_safe, one.delta, one.run_dir = "m2", lam, delta, str(run)
            train(one)
            summary = json.loads((run / "m2/training_summary.json").read_text(encoding="utf-8"))
            records.append({"lambda_safe": lam, "delta": delta,
                "validation_safe_ndcg_at_10": summary["best_validation_safe_ndcg_at_10"],
                "best_epoch": summary["best_epoch"], "run_dir": str(run)})
    result = pd.DataFrame(records).sort_values("validation_safe_ndcg_at_10", ascending=False)
    result.to_csv(out / "validation_sweep.csv", index=False)
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    def shared(p):
        p.add_argument("--root", default=str(ROOT_DEFAULT))
        p.add_argument("--health-condition", required=True,
                       help="One condition or a '|' separated combination; e.g. hypertension or diabetes|hypertension or comorbidity")
        p.add_argument("--seed", type=int, default=2020)
        p.add_argument("--embedding", type=int, default=64)
        p.add_argument("--distance", choices=("l1", "l2"), default="l1")
        p.add_argument("--margin", type=float, default=1.0)
        p.add_argument("--epochs", type=int, default=300)
        p.add_argument("--patience", type=int, default=20)
        p.add_argument("--batch-size", type=int, default=1024)
        p.add_argument("--learning-rate", type=float, default=0.001)
        p.add_argument("--device", default=None)
    tr = subs.add_parser("train")
    shared(tr)
    tr.add_argument("--run-dir", required=True)
    tr.add_argument("--model", choices=("m1", "m2"), required=True)
    tr.add_argument("--lambda-safe", type=float, default=0.5)
    tr.add_argument("--delta", type=float, default=1.0)
    ev = subs.add_parser("evaluate")
    ev.add_argument("--root", default=str(ROOT_DEFAULT))
    ev.add_argument("--health-condition", required=True,
                    help="Must match the condition used for training")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--device", default=None)
    sw = subs.add_parser("sweep")
    shared(sw)
    sw.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command in {"train", "sweep"} and (
            args.embedding <= 0 or args.epochs <= 0 or args.patience <= 0 or
            args.batch_size <= 0 or args.learning_rate <= 0 or args.margin <= 0):
        parser.error("embedding, epochs, patience, batch-size, learning-rate and margin must be positive")
    if args.command == "train" and args.model == "m2" and (args.lambda_safe <= 0 or args.delta < 0):
        parser.error("M2 needs lambda-safe > 0 and delta >= 0")
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        sweep(args)

if __name__ == "__main__":
    main()
