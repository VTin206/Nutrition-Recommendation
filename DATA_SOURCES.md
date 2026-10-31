# Bundled input provenance

These files are byte-for-byte copies from the parent research repository at packaging time:

- interactions/foodcom_train.csv
- interactions/foodcom_test.csv
- interactions/foodcom_train.txt
- interactions/foodcom_val.txt
- interactions/foodcom_test.txt
- recipes/foodcom_recipes_metadata.csv
- health/clinical_health_rules.json
- kg/foodcom_kg_triples.tsv
- latest version LightGCN_PathSim/model.py
- latest version LightGCN_PathSim/dataloader.py
- latest version LightGCN_PathSim/world.py

No synthetic users, diagnosis labels, nutrition values, or experiment results were added. `kg/foodcom_kg_triples.tsv` is copied from the parent research repository; its INTERACTED triples match the frozen training split, and it does not include validation/test interactions. `transe_model.py` implements the TransE score and margin objective for this package, following the TransE paper rather than copying an upstream KGE training loop. Food.com has no patient diagnosis labels, so each experiment run applies one explicitly selected health-condition scenario to every user; this is an experiment assumption, not observed patient data.
