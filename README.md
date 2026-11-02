# Food.com Safety Recommendation Experiment: TransE and RotatE

This folder is intended to be uploaded to Google Drive and run in Colab. It contains the fixed Food.com interaction splits, training knowledge graph, recipe nutrition metadata, configurable health rules, and TransE/RotatE training and evaluation code. The dataset has no patient diagnosis labels.

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
- The same M1/M2 experiment can run either **TransE** or **RotatE**. The bundled LightGCN files are retained for the shared data loader.

## TransE implementation and objectives

The TransE architecture, triple scoring, entity constraint, margin-ranking loss, and filtered negative sampler come from **PyKEEN 1.11.1**. This project does not implement TransE itself. PyKEEN provides a library implementation, not a pretrained Food.com checkpoint: the model is initialized and trained on the bundled Food.com training KG.

The triple score is the negative translation distance:

`s(h,r,t) = -||e_h + e_r - e_t||_1`

The distance can be L1 or L2. PyKEEN normalizes entity embeddings through TransE's built-in constraint after optimizer updates. PyKEEN's `BasicNegativeSampler` draws one corrupted head or tail per positive triple and filters known training triples. The project keeps an experiment loop so M2 can add its safety loss; the TransE model, KG margin loss, and negative sampler are library components.

**M1 — TransE baseline** uses only PyKEEN's margin-ranking loss:

`L_M1 = mean(max(0, margin + d_positive - d_negative))`

**M2 — Safety-aware TransE** uses the same PyKEEN TransE model, margin-ranking loss, and negative sampler, then adds the project-specific safety ranking loss:

`L_M2 = L_TransE + lambda_safe * mean(max(0, delta + s_unsafe - s_safe))`

The same selected health scenario and validation checkpoint-selection metric are used for M1 and M2. M1 has no safety term in its training loss. M2 learns from SAFE/UNSAFE recipe pairs. At final evaluation M1 ranks without filtering; M2 removes UNSAFE and UNKNOWN recipes after scoring and returns fewer than K if there are not enough SAFE recipes.

Because PyKEEN changes the model implementation and checkpoint format, train both models again in a new run folder. Checkpoints made by the previous hand-coded TransE implementation cannot be loaded by this version.

## RotatE from the original repository

RotatE is loaded from the original [KnowledgeGraphEmbedding repository](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding), pinned to commit `2e440e0f9c687314d5ff67ead68ce985dc446e3a`. The original `KGEModel`, `TrainDataset`, and `BidirectionalOneShotIterator` files are bundled under `vendor/KnowledgeGraphEmbedding/` with their MIT license. The adapter only maps Food.com users and recipes to KG triples; it does not reimplement RotatE scoring. The bundled source matches the original repository at the pinned commit.

RotatE M1 uses the original model's `train_step`, including its log-sigmoid objective, filtered head/tail negative sampling, frequency subsampling, and self-adversarial negative weights. RotatE M2 uses the same original RotatE model and sampler, then adds the project safety hinge loss in the same optimizer step. The adapter's `--embedding` value is the original `hidden_dim`; the upstream model doubles entity embedding width internally for complex-valued entities. Defaults are gamma 12, 128 negatives, adversarial temperature 1, Adam, and learning rate 0.0001. The safety loss and inference-time hard filter are project-specific additions.

Train M1 and M2 with the same architecture and run folder. Use a separate new folder for RotatE and retrain both arms; TransE and RotatE checkpoints are not interchangeable. Compare M1 against M2 within the same architecture. A direct TransE-versus-RotatE comparison also changes the base training objective and sampler, so report the architecture and training configuration.

## Evaluation

The evaluator reports only these metrics at K=10: Safety@K, Unsafe@1, Unknown@K, P-NDCG@K, NDCG@K, Precision@K, Recall@K, and RecommendationCoverage. The comparison table also identifies the health scenario, model, K, penalty, and number of evaluated users. Safety@K is undefined for users who receive fewer than 10 recipes. Unknown@K is the fraction of returned recipes whose safety status is UNKNOWN. RecommendationCoverage is the fraction of the recipe catalog recommended at least once; it measures item/catalog coverage, not the fraction of users reached. Recall@K is undefined when a user's relevant-item pool is empty.

P-NDCG uses gain `2^relevance - 1` for SAFE recipes and `2^(-P) - 1` for non-SAFE recipes. `P` is the nominal penalty; `2^(-P)-1` is the actual negative gain. The evaluator reports P-NDCG only for `P = 5`; charts and example comparisons also use `P = 5`. The ideal DCG uses SAFE relevant recipes from the same candidate set. A zero ideal DCG produces undefined P-NDCG, and negative scores are not clipped. Changing `P` only changes evaluation, so existing checkpoints can be reused without retraining.

The validation split has item IDs but no star ratings, so checkpoint selection uses binary preference NDCG@10: every held-out validation interaction is relevant, and no safety labels or safety filter are used. This same validation metric selects checkpoints for M1 and M2. M1 does not build safety labels or safety pairs during training; M2 uses them only for its safety loss. Test metrics use held-out 4/5-star ratings and apply the configured inference policy. Unobserved recipes used for negative sampling are training-time negatives, not confirmed dislikes.

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
RUN = str(OUTPUT_ROOT / 'runs' / 'transe_hypertension_pref_ndcg')
```

Install the dependencies from the cloned repository:

```bash
!pip install -q -r "$ROOT/requirements.txt"
```

The Colab runtime must use Python 3.11 or newer for the pinned PyKEEN version. RotatE source is included in this folder, so Colab does not need to clone another repository at runtime.

Train M1 and M2 separately with the same `HEALTH_CONDITION` and `RUN` path:

```bash
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN" --model m1 --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN" --model m2 --embedding 64 --distance l1 --margin 1.0 --lambda-safe 0.5 --delta 1.0 --epochs 300 --patience 20 --seed 2020
```

Then evaluate both models:

```bash
!python "$ROOT/experiment.py" evaluate --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN"
```

To run RotatE, set a separate output folder and pass `--architecture rotate` to both training cells. Train M1 and M2 separately, then evaluate; the evaluator reads the architecture from the run manifest:

```python
RUN_ROTATE = str(OUTPUT_ROOT / 'runs' / 'rotate_hypertension')
```

```bash
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN_ROTATE" --architecture rotate --model m1 --embedding 64 --gamma 12 --negative-sample-size 128 --adversarial-temperature 1 --epochs 300 --patience 20 --seed 2020
!python "$ROOT/experiment.py" train --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN_ROTATE" --architecture rotate --model m2 --embedding 64 --gamma 12 --negative-sample-size 128 --adversarial-temperature 1 --lambda-safe 0.5 --delta 1.0 --epochs 300 --patience 20 --seed 2020
!python "$ROOT/experiment.py" evaluate --root "$ROOT" --health-condition "$HEALTH_CONDITION" --run-dir "$RUN_ROTATE"
```

The condition used for evaluation must match the condition used in training. For another disease, change both `HEALTH_CONDITION` and `RUN` (for example `diabetes` and `transe_diabetes`) and train both models again. This keeps different condition scenarios in separate result folders. Use a new run folder if you train again; an existing run folder already contains checkpoints and a manifest.

Optional M2 validation sweep:

```bash
!python "$ROOT/experiment.py" sweep --root "$ROOT" --health-condition "$HEALTH_CONDITION" --output-dir "$OUTPUT_ROOT/runs/sweep_hypertension" --embedding 64 --distance l1 --margin 1.0 --epochs 300 --patience 20 --seed 2020
```

Each run manifest records the selected health condition and hashes the inputs and source code. The code and dataset are read from `/content/Nutrition-Recommendation`; checkpoints, CSV tables, and charts are saved to the Drive `RUN` folder.

## Outputs

Each model folder saves `best_checkpoint.pt`, `training_config.json`, `training_history.csv`, `loss_curves.png`, and `training_summary.json`. Evaluation saves per-user and aggregate tables containing only the selected metrics, recommendation examples, and charts for those metrics.

## Tests

Run the bundled unit tests with:

```bash
python -B -m unittest discover -s tests -v
```
