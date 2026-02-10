
# Model Card: Flux on bernoulli_glm

This document provides a summary of the `flux` model trained on the `bernoulli_glm` dataset.

## 1. Model & Pipeline

- **Model Architecture:** `flux`
- **Training Pipeline:** `Flow Matching`
- **Purpose:** Reconstruct posterior distributions in a Simulation-Based Inference (SBI) context.

## 2. Dataset

- **Dataset:** `bernoulli_glm`
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
| `depth` | `4` |
| `depth_single_blocks` | `8` |
| `val_emb_dim` | `10` |
| `id_emb_dim` | `10` |
| `qkv_bias` | `True` |
| `id_merge_mode` | `concat` |
| `theta` | `-1` |
| `params_dtype` | `bfloat16` |
| `id_embedding_strategy` | `['absolute', 'absolute']` |

## 4. Training Configuration

| Parameter | Value |
|---|---|
| `batch_size` | `256` |
| `nsteps` | `50000` |
| `ema_decay` | `0.999` |
| `multistep` | `1` |
| `early_stopping` | `False` |
| `val_every` | `100` |
| `experiment_id` | `1` |
| `restore_model` | `False` |
| `train_model` | `True` |
| `warmup_steps` | `500` |
| `decay_transition` | `0.8` |
| `rtol` | `0.0001` |
| `max_lr` | `0.0001` |
| `min_lr` | `1e-06` |

## 5. Evaluation

The model's performance is evaluated using the Classifier 2-Sample Test (C2ST). An accuracy score close to 0.5 indicates that the generated samples are highly similar to the true data distribution.

- **Average C2ST Accuracy:** 0.553 ± 0.015

---
*This model card was automatically generated.*
