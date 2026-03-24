# Best Model Configurations

Only showing parameters that vary across configuration versions for a given task/method. Rows show the best configuration version for each simulation budget.

## Task: two_moons

### Model: Flux1

#### Budget: 10k

| Method | Best Exp Version | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| diffusion_flux | v1 | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v8 | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | 1.3 | 500 |

#### Budget: 30k

| Method | Best Exp Version | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v5 | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 5 | 50000 | 1.1 | 100 |
| diffusion_flux | v2 | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 90000 | N/A | 100 |
| score_matching_flux | v6 | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |

#### Budget: 100k

| Method | Best Exp Version | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v2 | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |
| diffusion_flux | v2 | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |
| score_matching_flux | v3 | 8 | 16 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 3 | 50000 | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v5 | float32 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 5 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |

#### Budget: 30k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v3 | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 3 | 50000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v2 | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 90000 | - | - | N/A | 100 |

#### Budget: 100k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v2 | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v3 | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 3 | 50000 | - | - | N/A | 100 |

## Task: bernoulli_glm

### Model: Flux1

#### Budget: 10k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| diffusion_flux | v7 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v4 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | N/A | 100 |
| diffusion_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v4 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | N/A | 100 |

#### Budget: 100k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| diffusion_flux | v7 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | 1.3 | 500 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best Exp Version | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v5 | 16 | float32 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 5 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v7 | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 | - | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 1 | 50000 | True | False | N/A | 100 |
| diffusion_flux1joint | v1 | 8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v8 | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |

#### Budget: 100k

| Method | Best Exp Version | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v6 | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v6 | 8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v8 | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |

## Task: gaussian_linear

### Model: Flux1

#### Budget: 10k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| diffusion_flux | v2 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |
| score_matching_flux | v6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v4 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | N/A | 100 |
| diffusion_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v5 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 5 | 50000 | 1.1 | 100 |

#### Budget: 100k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| diffusion_flux | v6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| score_matching_flux | v2 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v7 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v5 | 8 | 8 | float32 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.999 | 5 | 50000 | - | - | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v4 | 16 | 10 | float32 | 10 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | False | 0.999 | 4 | 50000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v5 | 8 | 8 | float32 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.999 | 5 | 50000 | - | - | 1.1 | 100 |

#### Budget: 100k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | - | 4096 | True | 0.999 | 1 | 50000 | True | False | N/A | 100 |
| diffusion_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v2 | 16 | 10 | float32 | 10 | 0.6 | 0.0002 | 2e-06 | 1000 | - | 1024 | True | 0.999 | 2 | 100000 | - | - | N/A | 100 |

## Task: gaussian_mixture

### Model: Flux1

#### Budget: 10k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| diffusion_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v4 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | N/A | 100 |

#### Budget: 30k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | 1.3 | 500 |
| diffusion_flux | v4 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | N/A | 100 |
| score_matching_flux | v1 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |

#### Budget: 100k

| Method | Best Exp Version | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | 1.3 | 500 |
| diffusion_flux | v2 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |
| score_matching_flux | v2 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v1 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v1 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v4 | 16 | 10 | float32 | 10 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 4 | 50000 | - | - | N/A | 100 |
| score_matching_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |

#### Budget: 100k

| Method | Best Exp Version | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v3 | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 3 | 50000 | - | - | N/A | 100 |
| score_matching_flux1joint | v2 | 16 | 10 | float32 | 10 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | - | - | N/A | 100 |

## Task: slcp

### Model: Flux1

#### Budget: 10k

| Method | Best Exp Version | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| diffusion_flux | v7 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v6 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| diffusion_flux | v6 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | 1.1 | 100 |
| score_matching_flux | v1 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |

#### Budget: 100k

| Method | Best Exp Version | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| diffusion_flux | v1 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | 1.1 | 100 |
| score_matching_flux | v8 | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | 1.3 | 500 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v7 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v6 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 6 | 50000 | - | - | 1.1 | 100 |
| score_matching_flux1joint | v1 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 7 | 50000 | - | - | 1.1 | 100 |

#### Budget: 30k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |

#### Budget: 100k

| Method | Best Exp Version | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.experiment_id | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 8 | 100000 | - | - | 1.3 | 500 |
| score_matching_flux1joint | v2 | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 2 | 100000 | - | - | N/A | 100 |

