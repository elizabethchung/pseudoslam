# Archive: earlier experiments

These scripts are the model variants and scratch tests I tried on the way to the final model in [`../resnet_3.py`](../resnet_3.py). They are kept for reference and still run, but they are not part of the main pipeline.

Run them from the repository root after `pip install -e .`, for example `python pseudoslam/envs/archive/resnet_2obs.py`. They read and write data in `pseudoslam/envs/data/`, like the main scripts.

## Model variants (roughly in the order I tried them)

| File | What it tried |
|---|---|
| `cnn.py` | First baseline: a 1D CNN that classifies each LiDAR angle as obstacle or free space |
| `cnn_ray.py` | The same CNN with hyperparameter search using Ray Tune |
| `resnet.py` | 1D ResNet on the raw LiDAR scan |
| `resnet2.py`, `resnet_tsdf.py`, `resnet_tsdf_save.py` | First versions that add a TSDF branch next to the LiDAR branch |
| `resnet_multi_gated.py`, `resnet_gated.py`, `resnet_try.py` | Gating modules that decide how many obstacle slots are active |
| `resnet_counting.py`, `gating_model.py` | Predicting the number of obstacles first (a separate counting network) |
| `resnet_dynamic.py` | Variable obstacle counts without a gating module |
| `resnet_1obs.py`, `resnet_2obs.py` | Models trained on scenes with exactly one or two obstacles |
| `transfer_nn.py` | Transfer learning from the 1- and 2-obstacle models into the 3-obstacle model |
| `diverse_resnet_3.py`, `diverse_resnet_better.py` | The 3-obstacle model trained on a more varied dataset |
| `diverse_resnet_tsdf.py` | LiDAR-only network that uses the TSDF in the loss instead of as an input |

## Scratch tests

`test_func.py`, `test_loss.py`, `test_min_max.py`, `test_scan.py` and `test_tsdf.py` are quick checks I used while debugging the labels, loss, scan plotting and TSDF conversion. They are scripts, not a pytest suite.
