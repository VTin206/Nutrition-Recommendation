# Upstream RotatE implementation

The files in `codes/` are copied without modification from the official
KnowledgeGraphEmbedding repository by Sun et al.:

- Repository: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
- Pinned commit: `2e440e0f9c687314d5ff67ead68ce985dc446e3a`
- License: MIT (see `LICENSE`)

The experiment calls the upstream `KGEModel(model_name="RotatE")`,
`TrainDataset`, and `BidirectionalOneShotIterator`. `rotate_model.py` is an
adapter for the Food.com entity mappings and recommendation scoring; it does
not reimplement RotatE's scoring function.
