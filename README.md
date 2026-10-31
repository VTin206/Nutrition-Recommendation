# Food.com TransE Safety Recommendation Experiment

This folder is intended to be uploaded to Google Drive and run in Colab. It contains the fixed Food.com interaction splits, training knowledge graph, recipe nutrition metadata, configurable health rules, and TransE training/evaluation code. The dataset has no patient diagnosis labels.

## Health-condition setup

Choose one health condition for each experiment run. The code applies that same condition scenario to every user in that run. This recreates the older experiment setup without asking you to create a `profiles.csv` file.

Examples:

- `hypertension`
- `diabetes`
- `heart_failure`
- `fatty_liver`
- `kidney_failure`
- `comorbidity` for all five conditions together
- `diabetes|hypertension` for a selected combination

This is a run-wide research assumption, not a claim that Food.com users actually have that diagnosis. The configured nutrition rules are research assumptions and are not patient-specific medical advice. Missing nutrition values are marked UNKNOWN; the safety filter excludes both UNSAFE and UNKNOWN recipes.

The old per-user profile template is no longer needed and has been removed. You can train and evaluate without uploading any additional profile file.

## Data and model

- `kg/foodcom_kg_triples.tsv` contains user–recipe training interactions and recipe-side relations. Its `INTERACTED` triples match the frozen training split; validation and test interactions are excluded.
- The CSV interaction files provide preference labels for final test evaluation. Rating 4 maps to relevance 1 and rating 5 maps to relevance 2.
- This experiment trains **TransE only**. The bundled LightGCN files are retained for the shared data loader.

The triple score is the negative translation distance:

`s(h,r,t) = -||e_h + e_r - e_t||_1`

Entity embeddings are normalized to unit L2 length before each update. Training uses SGD, one uniformly corrupted head or tail per positive KG triple, rejects negatives already in the training KG, and uses TransE's margin-ranking objective.

**M1 — TransE baseline** uses only the TransE margin loss:

`L_M1 = mean(max(0, margin + d_positive - d_negative))`

**M2 — Safety-aware TransE** adds a safety ranking loss:

`L_M2 = L_TransE + lambda_safe * mean(max(0, delta + s_unsafe - s_safe))`

The same selected health scenario and validation checkpoint-selection metric are used for M1 and M2. M1 has no safety term in its training loss. M2 learns from SAFE/UNSAFE recipe pairs. At final evaluation M1 ranks without filtering; M2 removes UNSAFE and UNKNOWN recipes after scoring and returns fewer than K if there are not enough SAFE recipes.

## Evaluation

The evaluator reports metrics at K=5, 10, and 20: Safety@K, SafetyAmongReturned, FillRate@K, P-DCG/P-NDCG, NDCG, Precision, Recall, safety-conditioned Precision/Recall, Unsafe@1, Unknown@K, recommendation coverage, and sample counts. Recall is undefined when its relevant-item pool is empty. Safety@K is only reported for a full K-item list; short lists report SafetyAmongReturned and FillRate@K.

P-NDCG uses gain `2^relevance - 1` for SAFE recipes and `2^(-P) - 1` for non-SAFE recipes. `P` is the nominal penalty; `2^(-P)-1` is the actual negative gain. The ideal DCG uses SAFE relevant recipes from the same candidate set. A zero ideal DCG produces undefined P-NDCG, and negative scores are not clipped.

The validation split has item IDs but no star ratings, so checkpoint selection uses binary safe-conditioned NDCG@10. Test metrics use the held-out 4/5-star ratings. Unobserved recipes used for negative sampling are training-time negatives, not confirmed dislikes.

## Run in Colab

Open a new Colab notebook. Clone the repository so the code and bundled data are downloaded directly from Git:

```python
%cd /content
!git clone --depth 1 --branch main https://github.com/VTin206/Nutrition-Recommendation.git
ROOT = '/content/Nutrition-Recommendation'
```

Mount Drive only to keep the trained models and evaluation results after the Colab runtime ends:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path

ROOT = '/content/Nutrition-Recommendation'
OUTPUT_ROOT = Path('/content/drive/MyDrive/safety_aware_food_recommendation')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
HEALTH_CONDITION = 'hypertension'
RUN = str(OUTPUT_ROOT / 'runs' / 'transe_hypertension')
```

Install the dependencies from the cloned repository:

```bash
!pip install -q -r "$ROOT/requirements.txt"
```

Train M1 and M2 separately with the same `HEALTH_CONDITION` and `RUN` path:

```bash
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN" --model m1 --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN" --model m2 --embedding 64 --distance l1 --margin 1.0 --lambda-safe 0.5 --delta 1.0 --epochs 300 --patience 20 --seed 2020
```

Then evaluate both models:

```bash
!python "$ROOT/experiment.py" evaluate --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN"
```

The condition used for evaluation must match the condition used in training. For another disease, change both `HEALTH_CONDITION` and `RUN` (for example `diabetes` and `transe_diabetes`) and train both models again. This keeps different condition scenarios in separate result folders. Use a new run folder if you train again; an existing run folder already contains checkpoints and a manifest.

Optional M2 validation sweep:

```bash
!python "$ROOT/experiment.py" sweep --root "$ROOT" --health-condition "$HEALTH_CONDITION" --output-dir "$OUTPUT_ROOT/runs/sweep_hypertension" --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
```

Each run manifest records the selected health condition and hashes the inputs and source code. The code and dataset are read from `/content/Nutrition-Recommendation`; checkpoints, CSV tables, and charts are saved to the Drive `RUN` folder.

## Outputs

Each model folder saves `best_checkpoint.pt`, `training_config.json`, `training_history.csv`, `loss_curves.png`, and `training_summary.json`. Evaluation saves per-user and aggregate metric tables, recommendation examples, and charts for Safety@K, P-NDCG@K, NDCG@K, Recall@K, and FillRate@K.

## Tests

Run the bundled unit tests with:

```bash
python -B -m unittest discover -s tests -v
```
