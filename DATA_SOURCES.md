# Bundled input provenance

These files are byte-for-byte copies from the parent research repository at packaging time:

- interactions/foodcom_train.csv
- interactions/foodcom_test.csv
- interactions/foodcom_train.txt
- interactions/foodcom_val.txt
- interactions/foodcom_test.txt
- recipes/foodcom_recipes_metadata.csv
- health/clinical_health_rules.json
- latest version LightGCN_PathSim/model.py
- latest version LightGCN_PathSim/dataloader.py
- latest version LightGCN_PathSim/world.py

No synthetic users, diagnosis labels, nutrition values, or experiment results were added. The TransE scorer in transe_model.py is implemented for this experiment; it is not a copied upstream KGE training implementation. User diagnosis profiles must be supplied separately in profiles.csv.