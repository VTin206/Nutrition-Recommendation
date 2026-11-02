"""M1/M2 Food.com experiments with library-backed TransE and official RotatE."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pykeen.sampling import BasicNegativeSampler
from torch.utils.data import DataLoader
from safety_aware import (SafetyStatus, StatusLookup, assess_safety, dcg,
                          experiment_arms, parse_profile_conditions, prepare_metric_reference, safe_ndcg)

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = HERE
sys.path.insert(0, str(ROOT_DEFAULT / "latest version LightGCN_PathSim"))
from dataloader import Dataset, read_split
from transe_model import TransERecommender
from rotate_model import (RotatERecommender, TrainDataset as RotateTrainDataset,
                          BidirectionalOneShotIterator as RotateIterator,
                          UPSTREAM_COMMIT as ROTATE_UPSTREAM_COMMIT)

PYKEEN_VERSION = package_version("pykeen")
# The P=5-only evaluation update does not alter model training or checkpoints.
PRE_P5_EVALUATION_SOURCE_HASH = "2fe0a7929a6c1fa4b2f46ae28e53c746ded7476d2ed955ff3da39c3dd1096ede"
# The K=10-only evaluation update also leaves training and checkpoints unchanged.
PRE_K10_EVALUATION_SOURCE_HASH = "a67e721f90d629cd2cd4f33414635ba7939c47a314312d8cf041b448162f7b92"
# Restricting the reported metrics also changes evaluation output only.
PRE_SELECTED_METRICS_SOURCE_HASH = "d0ce62b0bc6a10ea407993e052bd47327433f51db2f8a771b763595224c37c35"

USER_METRICS = ["Safety@K", "Unsafe@1", "Unknown@K", "P-NDCG@K", "NDCG@K",
                "Precision@K", "Recall@K"]
REPORTED_METRICS = [*USER_METRICS, "RecommendationCoverage"]


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


def build_model(kg, embedding, distance="l1", margin=1.0, seed=2020):
    return TransERecommender(
        kg["triples"], kg["entity2id"], kg["relation2id"], embedding,
        kg["user_entity_ids"], kg["recipe_entity_ids"],
        kg["interacted_relation_id"], distance=distance, margin=margin, seed=seed)


def build_experiment_model(architecture, kg, embedding, distance, margin, gamma, seed):
    if architecture == "rotate":
        return RotatERecommender(
            nentity=len(kg["entity2id"]), nrelation=len(kg["relation2id"]),
            hidden_dim=embedding, gamma=gamma,
            user_entity_ids=kg["user_entity_ids"],
            recipe_entity_ids=kg["recipe_entity_ids"],
            interacted_relation_id=kg["interacted_relation_id"])
    if architecture == "transe":
        return build_model(kg, embedding, distance, margin, seed)
    raise ValueError(f"Unsupported KGE architecture: {architecture}")


def pair_scores(model, user_ids, item_ids):
    return model.score_pairs(user_ids, item_ids)


def score_user_batch(model, user_ids, item_chunk_size=1024):
    return model.score_all(user_ids, item_chunk_size=item_chunk_size)


def validation_score(model, dataset, val, raw_train, k=10):
    """Binary preference NDCG on validation positives, without safety filtering."""
    active, relevant_by_user = [], {}
    for user, heldout in val.items():
        relevant = set(map(int, heldout))
        if relevant:
            active.append(user)
            relevant_by_user[user] = relevant
    if not active:
        raise ValueError("Validation has no positive interactions; cannot select checkpoint.")
    model.eval()
    values_by_user = []
    with torch.no_grad():
        for start in range(0, len(active), 64):
            user_batch = active[start:start + 64]
            ids = torch.as_tensor(user_batch, dtype=torch.long, device=dataset.device)
            score_batch = score_user_batch(model, ids).cpu().numpy()
            for row, user in enumerate(user_batch):
                seen = np.zeros(dataset.m_items, dtype=bool)
                seen[raw_train.get(user, [])] = True
                candidates = np.flatnonzero(~seen)
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
               HERE / "transe_model.py", HERE / "experiment.py", HERE / "safety_aware.py",
               HERE / "rotate_model.py", HERE / "requirements.txt",
               HERE / "vendor/KnowledgeGraphEmbedding/codes/model.py",
               HERE / "vendor/KnowledgeGraphEmbedding/codes/dataloader.py",
               HERE / "vendor/KnowledgeGraphEmbedding/UPSTREAM.md",
               HERE / "vendor/KnowledgeGraphEmbedding/LICENSE"]
    return {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest(p)
            for p in paths + sources}


def make_rotate_training_iterator(kg, batch_size, negative_sample_size):
    """Use the upstream filtered sampler and alternating head/tail iterator."""
    triples = [tuple(map(int, row)) for row in kg["triples"]]
    common = dict(batch_size=batch_size, shuffle=True, num_workers=0,
                  collate_fn=RotateTrainDataset.collate_fn)
    head_loader = DataLoader(
        RotateTrainDataset(triples, len(kg["entity2id"]), len(kg["relation2id"]),
                           negative_sample_size, "head-batch"), **common)
    tail_loader = DataLoader(
        RotateTrainDataset(triples, len(kg["entity2id"]), len(kg["relation2id"]),
                           negative_sample_size, "tail-batch"), **common)
    return RotateIterator(head_loader, tail_loader), 2 * max(len(head_loader), len(tail_loader))


def official_rotate_constrained_step(model, optimizer, iterator, safety_users,
                                     safe_pool, unsafe_pool, weight, margin,
                                     temperature, rng, device):
    """Official RotatE objective plus the experiment's M2 safety hinge term."""
    model.train()
    positive, negative, subsampling_weight, mode = next(iterator)
    positive = positive.to(device)
    negative = negative.to(device)
    subsampling_weight = subsampling_weight.to(device)
    optimizer.zero_grad(set_to_none=True)

    negative_scores = model((positive, negative), mode=mode)
    negative_log = (
        F.softmax(negative_scores * temperature, dim=1).detach()
        * F.logsigmoid(-negative_scores)
    ).sum(dim=1)
    positive_log = F.logsigmoid(model(positive)).squeeze(1)
    positive_loss = -(subsampling_weight * positive_log).sum() / subsampling_weight.sum()
    negative_loss = -(subsampling_weight * negative_log).sum() / subsampling_weight.sum()
    kg_loss = (positive_loss + negative_loss) / 2

    count = len(positive)
    sampled_users = rng.choice(safety_users, size=count, replace=True)
    sampled_safe = np.asarray([rng.choice(safe_pool[int(user)]) for user in sampled_users],
                              dtype=np.int64)
    sampled_unsafe = np.asarray([rng.choice(unsafe_pool[int(user)]) for user in sampled_users],
                                dtype=np.int64)
    sampled_users = torch.as_tensor(sampled_users, dtype=torch.long, device=device)
    sampled_safe = torch.as_tensor(sampled_safe, dtype=torch.long, device=device)
    sampled_unsafe = torch.as_tensor(sampled_unsafe, dtype=torch.long, device=device)
    safe_scores = model.score_pairs(sampled_users, sampled_safe)
    unsafe_scores = model.score_pairs(sampled_users, sampled_unsafe)
    safety_loss = torch.relu(margin + unsafe_scores - safe_scores).mean()
    total_loss = kg_loss + weight * safety_loss
    if not torch.isfinite(total_loss):
        raise ValueError("Non-finite RotatE loss; stopping training.")
    total_loss.backward()
    optimizer.step()
    return {"kge_loss": float(kg_loss.detach()),
            "safety_loss": float(safety_loss.detach()),
            "total_loss": float(total_loss.detach()), "count": count}

def train(args):
    (root, metadata, rules, _, _, raw_train, val, _, paths, kg) = load_inputs(args.root)
    n_users = max(raw_train) + 1
    conditions, user_conditions = make_health_scenario(args.health_condition, rules, n_users)
    checkpoint_selector = "validation preference NDCG@10 (binary, no safety filtering)"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = Dataset(root / "interactions/foodcom_train.txt", len(metadata), device)
    dataset.validate_heldout(val)
    if args.model == "m2":
        user_keys, vectors = build_safety_index(metadata, rules, user_conditions, n_users)
        safe_positive, unsafe, safety_users = training_safety_pools(
            dataset, user_keys, vectors, raw_train)
        if not len(safety_users):
            raise ValueError("M2 needs safe training interactions and unsafe recipe candidates under this health scenario.")
    else:
        safe_positive, unsafe = {}, {}
        safety_users = np.empty(0, dtype=np.int64)
    seed_all(args.seed)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rotate = args.architecture == "rotate"
    architecture_label = "RotatE (official KnowledgeGraphEmbedding)" if rotate else "PyKEEN TransE"
    base_objective = ("Official KGEModel log-sigmoid objective with self-adversarial negatives"
                      if rotate else "PyKEEN MarginRankingLoss with sLCWA negatives")
    manifest = {
        "schema_version": 2,
        "architecture_id": args.architecture,
        "architecture": architecture_label,
        "implementation": ("vendor/KnowledgeGraphEmbedding/codes/model.py::KGEModel"
                           if rotate else "pykeen.models.TransE"),
        "upstream_repository": ("https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding"
                                if rotate else "https://pykeen.readthedocs.io/"),
        "upstream_commit": ROTATE_UPSTREAM_COMMIT if rotate else None,
        "pykeen_version": None if rotate else PYKEEN_VERSION,
        "training_objective": base_objective,
        "negative_sampler": ("Official TrainDataset filtered head/tail corruption with alternating iterator"
                             if rotate else "PyKEEN BasicNegativeSampler (filtered, head/tail)"),
        "health_condition": conditions,
        "distance": None if rotate else args.distance,
        "margin": None if rotate else args.margin,
        "embedding": args.embedding,
        "gamma": args.gamma if rotate else None,
        "negative_sample_size": args.negative_sample_size if rotate else None,
        "negative_adversarial_sampling": rotate,
        "adversarial_temperature": args.adversarial_temperature if rotate else None,
        "entity_count": len(kg["entity2id"]),
        "relation_count": len(kg["relation2id"]),
        "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "checkpoint_selector": checkpoint_selector,
        "input_sha256": input_hashes(root, paths),
    }
    manifest_path = run_dir / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Run folder inputs/settings differ. Choose a fresh run folder.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out = run_dir / args.model
    out.mkdir(exist_ok=False)
    model_config = {**manifest,
                    "model": args.model,
                    "training_objective": (base_objective if args.model == "m1" else
                        base_objective + " + project safety hinge loss"),
                    "lambda_safe": args.lambda_safe if args.model == "m2" else 0.,
                    "delta": args.delta if args.model == "m2" else 0.}
    (out / "training_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    model = build_experiment_model(
        args.architecture, kg, args.embedding, args.distance, args.margin,
        args.gamma, args.seed).to(device)
    if rotate:
        optimizer = torch.optim.Adam(model.kge.parameters(), lr=args.learning_rate)
        rotate_iterator, rotate_steps = make_rotate_training_iterator(
            kg, args.batch_size, args.negative_sample_size)
        official_args = argparse.Namespace(
            cuda=device.type == "cuda", negative_adversarial_sampling=True,
            adversarial_temperature=args.adversarial_temperature,
            uni_weight=False, regularization=0.0)
        triple_sampler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        rotate_iterator, rotate_steps, official_args = None, 0, None
        triple_sampler = BasicNegativeSampler(
            mapped_triples=torch.as_tensor(kg["triples"], dtype=torch.long),
            num_entities=len(kg["entity2id"]),
            num_relations=len(kg["relation2id"]),
            num_negs_per_pos=1,
            filtered=True,
            corruption_scheme=("head", "tail"),
        ).to(device)
    rng = np.random.default_rng(args.seed)
    safety_rng = np.random.default_rng(args.seed + 1)
    history, best, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, count, safety_count = {"kge_loss": 0., "safety_loss": 0., "total_loss": 0.}, 0, 0
        if rotate:
            for _ in range(rotate_steps):
                if args.model == "m1":
                    log = model.kge.train_step(model.kge, optimizer, rotate_iterator, official_args)
                    kg_loss = float(log["loss"])
                    safety_loss = 0.0
                    total_loss = kg_loss
                else:
                    log = official_rotate_constrained_step(
                        model, optimizer, rotate_iterator, safety_users,
                        safe_positive, unsafe, args.lambda_safe, args.delta,
                        args.adversarial_temperature, safety_rng, device)
                    kg_loss, safety_loss, total_loss = (
                        log["kge_loss"], log["safety_loss"], log["total_loss"])
                model.post_parameter_update()
                sums["kge_loss"] += kg_loss
                sums["safety_loss"] += safety_loss
                sums["total_loss"] += total_loss
                count += 1
                if args.model == "m2":
                    safety_count += 1
        else:
            order = rng.permutation(len(kg["triples"]))
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                batch = kg["triples"][indices]
                as_tensor = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
                positive_tensor = as_tensor(batch)
                negative_tensor, valid_negative_mask = triple_sampler.sample(positive_tensor)
                negative_tensor = negative_tensor.to(device)
                positive_scores = model.score_triples(*positive_tensor.unbind(dim=1)).unsqueeze(-1)
                if valid_negative_mask is None:
                    negative_scores = model.score_triples(
                        *negative_tensor.reshape(-1, 3).unbind(dim=1)
                    ).reshape(len(batch), -1)
                    kg_loss = model.kge.loss.process_slcwa_scores(
                        positive_scores=positive_scores, negative_scores=negative_scores)
                else:
                    valid_negative_mask = valid_negative_mask.to(device)
                    if not valid_negative_mask.any():
                        continue
                    negative_tensor = negative_tensor[valid_negative_mask]
                    negative_scores = model.score_triples(
                        *negative_tensor.unbind(dim=1))
                    kg_loss = model.kge.loss.process_slcwa_scores(
                        positive_scores=positive_scores,
                        negative_scores=negative_scores,
                        batch_filter=valid_negative_mask,
                    )
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
                model.post_parameter_update()
                n = len(batch)
                sums["kge_loss"] += float(kg_loss.detach()) * n
                sums["safety_loss"] += float(safe_loss.detach()) * safety_pairs
                safety_count += safety_pairs
                sums["total_loss"] += float(total.detach()) * n
                count += n
        if count == 0:
            raise ValueError("No training batches were processed in this epoch.")
        row = {"epoch": epoch, "kge_loss": sums["kge_loss"] / count,
               "safety_loss": sums["safety_loss"] / safety_count if safety_count else 0.0,
               "total_loss": sums["total_loss"] / count}
        row["validation_ndcg_at_10"] = validation_score(
            model, dataset, val, raw_train)
        selection_score = row["validation_ndcg_at_10"]
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
        "best_validation_ndcg_at_10": best,
        "epochs_completed": len(history), "health_condition": conditions,
        "scenario_users": n_users,
        "kg_triples": int(len(kg["triples"])),
        "training_users_with_safety_pairs": int(len(safety_users)), "device": str(device),
        "architecture": architecture_label,
        "upstream_commit": ROTATE_UPSTREAM_COMMIT if rotate else None,
        "pykeen_version": None if rotate else PYKEEN_VERSION,
        "training_objective": model_config["training_objective"]}, indent=2),
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
    for col, label in [("kge_loss", "KGE objective"), ("safety_loss", "Safety"),
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
    expected_hashes = manifest["input_sha256"]
    current_hashes = input_hashes(root, paths)
    changed_hashes = [name for name in set(expected_hashes) | set(current_hashes)
                      if expected_hashes.get(name) != current_hashes.get(name)]
    evaluation_only_code_change = (
        set(changed_hashes) == {"experiment.py"} and
        expected_hashes.get("experiment.py") in {
            PRE_P5_EVALUATION_SOURCE_HASH,
            PRE_K10_EVALUATION_SOURCE_HASH,
            PRE_SELECTED_METRICS_SOURCE_HASH,
        }
    )
    if changed_hashes and not evaluation_only_code_change:
        raise ValueError(
            "Evaluation data/code/dependency hashes differ from training. "
            "Train fresh checkpoints in a new run folder with the current implementation."
        )
    if manifest.get("health_condition") != conditions:
        raise ValueError("Evaluation health condition differs from training. Use the same --health-condition.")
    architecture = manifest.get("architecture_id", "transe")
    if architecture == "transe" and manifest.get("pykeen_version") != PYKEEN_VERSION:
        raise ValueError(
            f"Checkpoint was trained with PyKEEN {manifest.get('pykeen_version')}; "
            f"the current runtime has {PYKEEN_VERSION}. Install the pinned requirements."
        )
    if architecture == "rotate" and manifest.get("upstream_commit") != ROTATE_UPSTREAM_COMMIT:
        raise ValueError("Checkpoint RotatE source commit differs from this package's pinned upstream source.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = Dataset(root / "interactions/foodcom_train.txt", len(metadata), device)
    dataset.validate_heldout(test)
    arms = experiment_arms()
    available_models = {
        model_name for _, model_name, _ in arms
        if (run_dir / model_name / "best_checkpoint.pt").is_file()
    }
    if not available_models:
        raise FileNotFoundError(f"No trained {manifest['architecture']} checkpoint found under {run_dir}.")
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
            model = build_experiment_model(
                architecture, kg, manifest["embedding"], manifest.get("distance") or "l1",
                manifest["margin"], manifest.get("gamma") or 12.0,
                manifest["seed"]).to(device)
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
                    for rank, item in enumerate(unfiltered[:10], 1):
                        if states[item] != SafetyStatus.SAFE.value:
                            examples.append({"user_id": user, "kind": "baseline_non_safe_top10",
                                "item_id": item, "item_name": metadata.iloc[item].get("name", ""),
                                "safety_status": states[item], "score": float(scores[user, item]),
                                "baseline_rank": rank})
            if hard_filter and any(states[i] != SafetyStatus.SAFE.value for i in ranked[:10]):
                raise AssertionError("Hard filter emitted a non-SAFE item.")
            for k in (10,):
                recs = ranked[:k]
                base = safe_ndcg(recs, relevance.get(user, {}), user_status, candidates, k, 5, reference)
                rec_rel = [float(relevance.get(user, {}).get(i, 0.0)) for i in recs]
                safe_flags = [user_status.get(i, SafetyStatus.UNKNOWN) == SafetyStatus.SAFE for i in recs]
                for penalty in (5,):
                    metrics = dict(base)
                    metrics["P-DCG@K"] = dcg(rec_rel, k, safe_flags, penalty)
                    metrics["P-NDCG@K"] = (metrics["P-DCG@K"] / base["P_IDCG@K"]
                        if base["P_IDCG@K"] > 0 else None)
                    returned = metrics["RecommendationCount"]
                    metrics["Unknown@K"] = (
                        metrics["Unknown@K"] / returned if returned else None
                    )
                    rows.append({"health_condition": "|".join(conditions),
                        "system": label, "model": model_name, "hard_filter": hard_filter,
                        "user_id": user, "K": k, "penalty_P": penalty, "catalog_size": dataset.m_items,
                        "coverage_status": ("FULL_TOP_K" if len(recs) == k else
                            ("INSUFFICIENT_SAFE_CANDIDATES" if hard_filter else "INSUFFICIENT_CANDIDATES")),
                        "recommended": recs,
                        "safe_returned_count": sum(user_status.get(i) == SafetyStatus.SAFE for i in recs),
                        **metrics})
        print(f"Evaluated {label}", flush=True)
    per_user_columns = ["health_condition", "system", "model", "hard_filter",
                        "user_id", "K", "penalty_P", *USER_METRICS]
    pd.DataFrame(rows)[per_user_columns].to_csv(run_dir / "per_user_metrics.csv", index=False)
    table = aggregate_rows(rows)
    table.to_csv(run_dir / "model_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(run_dir / "recommendation_examples.csv", index=False)
    comparison_examples(rows, run_dir)
    plot_metrics(table, run_dir)
    print(table.to_string(index=False))

def aggregate_rows(rows):
    frame = pd.DataFrame(rows)
    group_cols = ["health_condition", "system", "model", "hard_filter", "K", "penalty_P"]
    output = []
    for key, part in frame.groupby(group_cols, dropna=False, sort=False):
        record = dict(zip(group_cols, key))
        record["users_evaluated"] = int(part.user_id.nunique())
        for metric in USER_METRICS:
            record[metric] = part[metric].mean() if part[metric].notna().any() else np.nan
        distinct = {item for recs in part.recommended for item in recs}
        record["RecommendationCoverage"] = len(distinct) / int(part.catalog_size.iloc[0])
        output.append(record)
    return pd.DataFrame(output, columns=[*group_cols, "users_evaluated", *REPORTED_METRICS])


def plot_metrics(table, folder):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; saved comparison tables only.", flush=True)
        return
    charts = [("Safety@K", "safety_at_k.png"), ("Unsafe@1", "unsafe_at_1.png"),
        ("Unknown@K", "unknown_at_k.png"), ("P-NDCG@K", "p_ndcg_at_k.png"),
        ("NDCG@K", "ndcg_at_k.png"), ("Precision@K", "precision_at_k.png"),
        ("Recall@K", "recall_at_k.png"),
        ("RecommendationCoverage", "recommendation_coverage.png")]
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
    """Sweep lambda/delta, selecting checkpoints by preference-only validation NDCG."""
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
                "validation_ndcg_at_10": summary["best_validation_ndcg_at_10"],
                "best_epoch": summary["best_epoch"], "run_dir": str(run)})
    result = pd.DataFrame(records).sort_values("validation_ndcg_at_10", ascending=False)
    result.to_csv(out / "validation_sweep.csv", index=False)
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    def shared(p):
        p.add_argument("--root", default=str(ROOT_DEFAULT))
        p.add_argument("--health-condition", required=True,
                       help="One condition or a '|' separated combination; e.g. hypertension or diabetes|hypertension or comorbidity")
        p.add_argument("--architecture", choices=("transe", "rotate"), default="transe",
                       help="KGE architecture: PyKEEN TransE or RotatE from the vendored original repository")
        p.add_argument("--seed", type=int, default=2020)
        p.add_argument("--embedding", type=int, default=64,
                       help="TransE embedding dimension; RotatE upstream hidden dimension (entity vectors are twice this size)")
        p.add_argument("--distance", choices=("l1", "l2"), default="l1")
        p.add_argument("--margin", type=float, default=1.0)
        p.add_argument("--gamma", type=float, default=12.0,
                       help="RotatE upstream gamma parameter")
        p.add_argument("--negative-sample-size", type=int, default=128,
                       help="RotatE upstream filtered negatives per positive triple")
        p.add_argument("--adversarial-temperature", type=float, default=1.0,
                       help="RotatE upstream self-adversarial sampling temperature")
        p.add_argument("--epochs", type=int, default=300)
        p.add_argument("--patience", type=int, default=20)
        p.add_argument("--batch-size", type=int, default=1024)
        p.add_argument("--learning-rate", type=float, default=None,
                       help="Defaults to 0.001 for TransE and the original RotatE default 0.0001")
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
    if args.command in {"train", "sweep"} and args.learning_rate is None:
        args.learning_rate = 0.0001 if args.architecture == "rotate" else 0.001
    if args.command in {"train", "sweep"} and (
            args.embedding <= 0 or args.epochs <= 0 or args.patience <= 0 or
            args.batch_size <= 0 or args.learning_rate <= 0 or args.margin <= 0 or
            args.gamma <= 0 or args.negative_sample_size <= 0 or
            args.adversarial_temperature <= 0):
        parser.error("embedding, epochs, patience, batch-size, learning-rate, margin, gamma, negative-sample-size, and adversarial-temperature must be positive")
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
