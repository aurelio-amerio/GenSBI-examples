# Best Model Configurations

Only showing parameters that vary across configuration versions for a given task/method. Rows show the best configuration version for each training method.

## Task: two_moons

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 (0.549) | v8 (0.552) | concat | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v1 (0.554) | v6 (0.559) | sum | 8 | 16 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux | v8 (0.529) | v5 (0.531) | concat | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v5 (0.538) | v6 (0.540) | concat | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v2 (0.534) | v1 (0.543) | sum | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 90000 | False | True | N/A | 100 |
| score_matching_flux | v6 (0.517) | v5 (0.518) | concat | 4 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth | model.depth_single_blocks | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v2 (0.525) | v3 (0.528) | sum | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |
| diffusion_flux | v2 (0.523) | v3 (0.541) | sum | 8 | 16 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |
| score_matching_flux | v3 (0.507) | v2 (0.508) | sum | 8 | 16 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | False | True | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v5 (0.546) | v6 (0.548) | concat | float32 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v8 (0.613) | v2 (0.638) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v8 (0.522) | v4 (0.524) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v3 (0.529) | v2 (0.529) | concat | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 (0.603) | v2 (0.614) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v2 (0.513) | v8 (0.519) | concat | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 90000 | False | True | N/A | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v2 (0.524) | v3 (0.524) | concat | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 (0.605) | v2 (0.616) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v3 (0.502) | v2 (0.504) | concat | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | False | True | N/A | 100 |

## Task: bernoulli_glm

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 (0.722) | v5 (0.734) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v7 (0.701) | v5 (0.727) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux | v1 (0.692) | v4 (0.721) | sum | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v4 (0.696) | v7 (0.713) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |
| diffusion_flux | v1 (0.673) | v4 (0.682) | sum | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux | v4 (0.579) | v7 (0.589) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 (0.550) | v1 (0.557) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v7 (0.575) | v4 (0.576) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux | v8 (0.546) | v6 (0.554) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.616) | v7 (0.645) | concat | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v5 (0.824) | v6 (0.851) | concat | 16 | float32 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux1joint | v7 (0.630) | v4 (0.662) | concat | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 (0.578) | v6 (0.583) | concat | - | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | True | False | N/A | 100 |
| diffusion_flux1joint | v1 (0.776) | v6 (0.841) | concat | 16 | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux1joint | v8 (0.576) | v1 (0.584) | concat | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v6 (0.555) | v4 (0.556) | concat | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v6 (0.794) | v2 (0.810) | concat | 8 | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux1joint | v8 (0.560) | v2 (0.571) | concat | - | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

## Task: gaussian_linear

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 (0.714) | v3 (0.761) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v2 (0.718) | v3 (0.729) | sum | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |
| score_matching_flux | v6 (0.666) | v7 (0.820) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v4 (0.508) | v6 (0.737) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |
| diffusion_flux | v1 (0.526) | v5 (0.742) | sum | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux | v5 (0.509) | v6 (0.512) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v1 (0.504) | v4 (0.506) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 10000 | True | False | N/A | 100 |
| diffusion_flux | v6 (0.523) | v1 (0.524) | concat | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux | v2 (0.500) | v5 (0.500) | sum | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.564) | v7 (0.567) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v7 (0.707) | v8 (0.773) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux1joint | v5 (0.508) | v6 (0.512) | concat | 8 | 8 | float32 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v4 (0.512) | v7 (0.514) | concat | 16 | 10 | float32 | 10 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |
| diffusion_flux1joint | v8 (0.600) | v7 (0.674) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v5 (0.503) | v7 (0.508) | concat | 8 | 8 | float32 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | - | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | strategy.method | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 (0.499) | v4 (0.500) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | - | 4096 | True | 0.999 | 50000 | True | False | N/A | 100 |
| diffusion_flux1joint | v8 (0.601) | v7 (0.651) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | diffusion | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v2 (0.501) | v3 (0.503) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0002 | 2e-06 | 1000 | - | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |

## Task: gaussian_mixture

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 (0.553) | v4 (0.554) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v1 (0.562) | v6 (0.566) | sum | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux | v4 (0.524) | v6 (0.525) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v8 (0.520) | v6 (0.526) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux | v4 (0.520) | v6 (0.522) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |
| score_matching_flux | v1 (0.515) | v6 (0.516) | sum | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v8 (0.510) | v2 (0.516) | sum | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux | v2 (0.510) | v6 (0.516) | sum | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |
| score_matching_flux | v2 (0.502) | v3 (0.507) | sum | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.522) | v4 (0.527) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v1 (0.558) | v4 (0.559) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux1joint | v1 (0.510) | v4 (0.512) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.515) | v2 (0.521) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v4 (0.550) | v1 (0.553) | concat | 16 | 10 | float32 | 10 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | False | 0.999 | 50000 | False | True | N/A | 100 |
| score_matching_flux1joint | v8 (0.504) | v1 (0.508) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.params_dtype | model.val_emb_dim | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.513) | v2 (0.514) | concat | 8 | 8 | bfloat16 | 8 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v3 (0.543) | v2 (0.550) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | True | 0.999 | 50000 | False | True | N/A | 100 |
| score_matching_flux1joint | v2 (0.501) | v3 (0.503) | concat | 16 | 10 | float32 | 10 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |

## Task: slcp

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 (0.833) | v6 (0.841) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v7 (0.848) | v6 (0.855) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux | v6 (0.846) | v7 (0.854) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v6 (0.725) | v7 (0.803) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v6 (0.747) | v5 (0.808) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux | v1 (0.692) | v6 (0.739) | concat | 4 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.num_heads | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux | v7 (0.602) | v4 (0.619) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux | v1 (0.617) | v7 (0.676) | concat | 4 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |
| score_matching_flux | v8 (0.678) | v2 (0.692) | concat | 6 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v7 (0.657) | v8 (0.714) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| diffusion_flux1joint | v6 (0.826) | v7 (0.871) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.999 | 50000 | False | True | 1.1 | 100 |
| score_matching_flux1joint | v1 (0.663) | v4 (0.677) | concat | float32 | 0.6 | 0.0004 | 4e-06 | 500 | 4096 | False | 0.999 | 50000 | True | False | N/A | 100 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.586) | v4 (0.596) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v8 (0.680) | v6 (0.684) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v8 (0.686) | v1 (0.716) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | optimizer.warmup_steps | training.batch_size | training.early_stopping | training.ema_decay | training.nsteps | training.restore_model | training.train_model | training.val_error_ratio | training.val_every |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.566) | v1 (0.569) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| diffusion_flux1joint | v8 (0.655) | v6 (0.665) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 500 | 256 | True | 0.9999 | 100000 | False | True | 1.3 | 500 |
| score_matching_flux1joint | v2 (0.534) | v3 (0.601) | concat | float32 | 0.6 | 0.0002 | 2e-06 | 1000 | 1024 | True | 0.999 | 100000 | False | True | N/A | 100 |

