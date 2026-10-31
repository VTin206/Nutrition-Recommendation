# Food.com TransE Safety Recommendation Experiment

This folder is self-contained for Google Colab. It includes the frozen Food.com train/validation/test splits, the training knowledge graph, recipe nutrition metadata, configurable health rules, and the TransE training/evaluation code. User diagnosis profiles remain user-supplied because the dataset has no patient-level condition labels.

## Data and limits

- `kg/foodcom_kg_triples.tsv` contains the training KG: user–recipe interactions plus recipe-side relations. Its `INTERACTED` triples match the frozen training split exactly; validation and test interactions are kept out of the KG.
- The interaction CSVs contain the preference labels used for final evaluation. The KG includes only training interactions.
- `profiles_template.csv` is a template, not real diagnosis data. Blank conditions mean UNKNOWN; enter `none` only for a verified no-disease profile.
- Nutritional thresholds are configurable research assumptions, not independently clinically validated rules. Missing nutrition values are UNKNOWN.
- The older LightGCN source folder is retained because its data loader reads the frozen interaction splits. This experiment trains **TransE only**.

## TransE training objectives

The score of a triple is the negative translation distance:

`s(h,r,t) = -||e_h + e_r - e_t||_1`

The implementation initializes embeddings using the TransE paper's range, normalizes entity vectors to unit L2 norm before each update, and uses SGD with one uniformly corrupted head or tail per positive KG triple. Corrupted triples already present in the training KG are rejected. The held-out validation/test triples are not consulted by the negative sampler.

**M1 — TransE baseline** uses only the mean margin-ranking objective:

`L_TransE = mean(max(0, margin + d_positive - d_negative))`

It does not use BPR or a safety loss. This is TransE's KG-triple training objective, rather than a recommender-specific BPR objective. For Top-K recommendation, the trained model scores `(User, INTERACTED, Recipe)` triples and ranks recipes by score.

**M2 — Safety-aware TransE** adds the separately configurable safety ranking term:

`L_total = L_TransE + lambda_safe * mean(max(0, delta + s_unsafe - s_safe))`

The safety term uses configured profile/rule labels. UNKNOWN is not silently treated as SAFE. M2 is a safety-aware extension, not the original TransE objective.

The training split, KG, initialization seed, and TransE margin objective are shared by M1 and M2. Validation safe-conditioned NDCG@10 selects the checkpoint; test data is not used for checkpoint selection. At evaluation, M1 ranks without a safety filter. M2 applies a deterministic hard filter after training: only SAFE recipes are returned, in their TransE score order. UNSAFE and UNKNOWN recipes are excluded, and short lists are not backfilled.

## Evaluation

At K=5, 10, and 20, the evaluator reports preference NDCG/Precision/Recall, safety-conditioned Precision/Recall, Safety@K, SafetyAmongReturned, FillRate@K, Unsafe@1, Unknown@K, recommendation coverage, P-DCG/P-NDCG, and sample counts. Recall is undefined when the relevant pool is empty. Safety@K is reported only for lists with exactly K returned recipes; shorter lists report SafetyAmongReturned and FillRate@K.

P-NDCG uses gain `2^relevance - 1` for SAFE recipes and `2^(-P) - 1` for non-SAFE recipes. P is the nominal penalty, while `2^(-P)-1` is the actual negative gain. Eligible IDCG uses SAFE relevant recipes from the same candidate universe. A zero IDCG produces undefined P-NDCG; negative scores are not clipped.

The validation split contains item IDs but no star ratings, so checkpoint selection uses binary safe-conditioned NDCG. Test metrics use graded relevance: rating 4 maps to 1 and rating 5 maps to 2. Unobserved recipes used in negative sampling are training-time negative samples, not confirmed dislikes.

## Run in Colab

Upload this entire folder to Drive, mount Drive, and set its path:

```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/safety_aware_food_recommendation'
PROFILES = ROOT + '/profiles.csv'
RUN = ROOT + '/runs/transe_margin'
```

Install the package requirements if the Colab runtime does not already provide them:

```bash
!pip install -q -r "$ROOT/requirements.txt"
```

Copy `profiles_template.csv` to `profiles.csv` and enter the profiles you actually have. Train the pure TransE baseline and the safety-aware TransE variant separately:

```bash
!python "$ROOT/experiment.py" train --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN" --model m1 --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
!python "$ROOT/experiment.py" train --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN" --model m2 --embedding 64 --distance l1 --margin 1.0 --lambda-safe 0.5 --delta 1.0 --epochs 300 --patience 20 --seed 2020
```

Evaluate both trained variants in a separate step:

```bash
!python "$ROOT/experiment.py" evaluate --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN"
```

The evaluation compares exactly M1 and M2. Train both checkpoints before evaluating to get the direct comparison; if only one checkpoint exists, only that method is reported.

Run the validation-only safety hyperparameter sweep with:

```bash
!python "$ROOT/experiment.py" sweep --root "$ROOT" --profiles "$PROFILES" --output-dir "$ROOT/runs/sweep" --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
```

Use a new run directory for retraining with changed data or settings. The manifest hashes the inputs and relevant source files and rejects mismatched evaluation inputs.

## Outputs

Each trained variant saves `best_checkpoint.pt`, `training_config.json`, `training_history.csv`, `loss_curves.png`, and `training_summary.json`. Evaluation saves per-user and aggregate metric tables, user-level recommendation examples, baseline unsafe examples, and charts for Safety@K, P-NDCG@K, NDCG@K, Recall@K, and FillRate@K.

## Tests

```bash
python -B -m unittest discover -s "$ROOT/tests" -v
```
