
# Model Card: Simformer on two_moons

This document provides a summary of the `simformer` model trained on the `two_moons` dataset.

## 1. Model & Pipeline

- **Model Architecture:** `simformer`
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
| `dim_value` | `40` |
| `dim_id` | `40` |
| `dim_condition` | `10` |
| `fourier_features` | `128` |
| `num_heads` | `6` |
| `num_layers` | `8` |
| `widening_factor` | `3` |
| `qkv_features` | `60` |
| `num_hidden_layers` | `1` |

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
| `patience` | `10` |
| `cooldown` | `2` |
| `factor` | `0.5` |
| `accumulation_size` | `100` |
| `rtol` | `0.0001` |
| `max_lr` | `0.001` |
| `min_lr` | `1e-08` |

## 5. Evaluation

The model's performance is evaluated using the Classifier 2-Sample Test (C2ST). An accuracy score close to 0.5 indicates that the generated samples are highly similar to the true data distribution.

- **Average C2ST Accuracy:** 0.528 ± 0.035

---
*This model card was automatically generated.*
