# Safety-Aware Food Recommendation Experiment

This folder is self-contained for Google Colab: it includes frozen Food.com splits, recipe nutrition metadata, configurable health rules, a TransE recommendation scorer, and the optional repository LightGCN source. The diagnosis profile remains user-supplied because the repository has no patient-level condition data.

## Repository inspection and limits

- Default backbone: TransE over user--INTERACTED-->recipe triples. The optional LightGCN backbone uses the repository's three-layer implementation.
- Included data: train/test interaction CSVs, train/validation/test text splits, recipe metadata, and health rule JSON.
- No user diagnosis profiles or serving-size normalization are included. Do not infer diagnoses from interactions.
- Existing nutrition thresholds are configurable research assumptions; this experiment does not claim independent clinical validation. Missing required nutrient values become UNKNOWN.

## Models and metrics

TransE scores a user/recipe pair as s(u,i) = gamma - ||e_user + r_INTERACTED - e_recipe||_1, with gamma fixed at 12. Because gamma is an additive constant, it cancels from the pairwise BPR and safety losses and is not tuned. It uses BPR over user-recipe interactions. This is a TransE-based recommender, not the full official KGE training loop: the Colab bundle does not include auxiliary KG triples or the official entity-negative-sampling pipeline. LightGCN remains available with --backbone lightgcn.

M1 optimizes BPR preference loss on safe train positives versus safe train-unobserved items. Four/five-star test interactions use graded relevance 1/2 for ranking evaluation; training BPR treats train ratings >=4 as positive interactions. The training sampler does not inspect validation or test labels. Eligible items absent from train are treated as implicit BPR negatives, not observed dislikes; a future held-out positive can therefore be sampled as an unknown item. M2 adds:
L_total = L_pref + lambda_safe * mean(max(0, delta + s_unsafe - s_safe)).

M1 and M2 share the selected backbone, profile-eligible train positives, preference samples, split, seed, and maximum budget. LightGCN propagates over the safe-filtered training graph; TransE scores user-recipe interaction triples. M1+Filter and M2+Filter remove UNSAFE and UNKNOWN items at inference, preserve ranking order, and do not backfill short lists.

At K=5,10,20 the evaluator reports standard NDCG, preference Precision/Recall, eligible Precision/Recall, Safety@K, SafetyAmongReturned, FillRate@K, Unsafe@1, Unknown@K, coverage, P-DCG/P-NDCG, and sample counts. Recall is undefined for an empty relevant pool. Safety@K counts only confirmed SAFE items in its numerator, so UNKNOWN contributes zero and is also reported separately; the metric is reported only for an exactly K-length list.

P-NDCG uses gain 2^relevance-1 for SAFE and 2^(-P)-1 for UNSAFE or UNKNOWN. P is the nominal penalty; 2^(-P)-1 is the actual negative gain. Eligible IDCG uses safe items from the same candidate universe. Zero IDCG yields undefined P-NDCG; negative scores are preserved without clipping. UNKNOWN is counted separately and is not asserted clinically unsafe.

Checkpoint selection uses validation safe-conditioned NDCG@10 with binary labels because the frozen validation text split contains item IDs but not ratings. Test NDCG uses graded preference relevance (rating 4 maps to 1; rating 5 maps to 2). Lambda/delta selection uses validation only. P=[1,3,5,10] is an evaluation sensitivity and does not retrain. Example recommendations are illustrative, not aggregate evidence.

## Colab setup

Upload this entire folder to Drive and mount it. Copy profiles_template.csv to profiles.csv, then enter actual profile data. Separate conditions with |. Blank is UNKNOWN; use none only for a verified no-disease profile.

    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/safety_aware_food_recommendation'
    PROFILES = ROOT + '/profiles.csv'
    RUN = ROOT + '/runs/main'

Optional dependencies installation:
    !pip install -r "$ROOT/requirements.txt"

Train each model separately:
    !python "$ROOT/experiment.py" train --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN" --model m1 --backbone transe --epochs 300 --patience 20 --seed 2020
    !python "$ROOT/experiment.py" train --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN" --model m2 --backbone transe --lambda-safe 0.5 --delta 1.0 --epochs 300 --patience 20 --seed 2020
    !python "$ROOT/experiment.py" evaluate --root "$ROOT" --profiles "$PROFILES" --run-dir "$RUN"

Validation-only lambda/delta sweep:
    !python "$ROOT/experiment.py" sweep --root "$ROOT" --profiles "$PROFILES" --output-dir "$ROOT/runs/sweep" --backbone transe --epochs 300 --patience 20 --seed 2020

## Outputs

Each model saves its best checkpoint, loss history CSV/plot, and validation summary. Evaluation saves per_user_metrics.csv, model_comparison.csv, user_level_examples.csv, recommendation_examples.csv, and charts for Safety@K, P-NDCG@K, NDCG@K, Recall@K, and FillRate@K. Inputs and code are SHA-256 recorded in the run manifest.

## Tests

    python -B -m unittest discover -s "$ROOT/tests" -v