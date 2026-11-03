# Bundled input provenance

These dataset files are byte-for-byte copies from the parent research repository at packaging time:

- interactions/foodcom_train.csv
- interactions/foodcom_test.csv
- interactions/foodcom_train.txt
- interactions/foodcom_val.txt
- interactions/foodcom_test.txt
- recipes/foodcom_recipes_metadata.csv
- health/clinical_health_rules.json
- kg/foodcom_kg_triples.tsv

No synthetic users, diagnosis labels, nutrition values, or experiment results were added. `kg/foodcom_kg_triples.tsv` is copied from the parent research repository; its INTERACTED triples match the frozen training split, and it does not include validation/test interactions. `foodcom_data.py` contains the project-local parser and split-validation helpers used by the experiment. `pathsim_model.py` implements the PathSim formula from Sun et al. (2011) with SciPy sparse matrix operations; it is not copied from a third-party repository. TransE scoring, the margin-ranking loss, and filtered negative sampling are provided by PyKEEN 1.11.1; `transe_model.py` only adapts the library model to this project's user/recipe IDs. RotatE's `KGEModel`, `TrainDataset`, and `BidirectionalOneShotIterator` are copied unmodified from the official KnowledgeGraphEmbedding repository, pinned at commit `2e440e0f9c687314d5ff67ead68ce985dc446e3a`; `rotate_model.py` only adapts Food.com IDs to the original API. Food.com has no patient diagnosis labels, so each experiment run applies one explicitly selected health-condition scenario to every user; this is an experiment assumption, not observed patient data.
