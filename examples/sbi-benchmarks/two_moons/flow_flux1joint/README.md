
# Model Card: Flux1joint on two_moons

This document provides a summary of the `flux1joint` model trained on the `two_moons` dataset.

## 1. Model & Pipeline

- **Model Architecture:** `flux1joint`
- **Training Pipeline:** `Flow Matching`
- **Purpose:** Reconstruct posterior distributions in a Simulation-Based Inference (SBI) context.

## 2. Dataset

- **Dataset:** `two_moons`
- **Description:** A synthetic benchmark dataset.
- **Training Size:** The model was trained on 100,000 (1e5) samples.

## 3. Model Architecture

| Parameter | Value |
|---|---|
| `in_channels` | `1` |
| `vec_in_dim` | `None` |
| `context_in_dim` | `1` |
| `mlp_ratio` | `4` |
| `num_heads` | `4` |
| `depth_single_blocks` | `16` |
| `val_emb_dim` | `10` |
| `cond_emb_dim` | `4` |
| `id_emb_dim` | `10` |
| `id_merge_mode` | `concat` |
| `qkv_bias` | `True` |
| `params_dtype` | `float32` |

## 4. Training Configuration

| Parameter | Value |
|---|---|
| `batch_size` | `4096` |
| `nsteps` | `50000` |
| `ema_decay` | `0.999` |
| `multistep` | `1` |
| `early_stopping` | `True` |
| `val_every` | `100` |
| `experiment_id` | `1` |
| `restore_model` | `True` |
| `train_model` | `False` |
| `warmup_steps` | `500` |
| `decay_transition` | `0.6` |
| `rtol` | `0.0001` |
| `max_lr` | `0.0004` |
| `min_lr` | `4e-06` |

## 5. Evaluation

The model's performance is evaluated using the Classifier 2-Sample Test (C2ST). An accuracy score close to 0.5 indicates that the generated samples are highly similar to the true data distribution.

- **Average C2ST Accuracy:** 0.503 ± 0.010

---
*This model card was automatically generated.*
