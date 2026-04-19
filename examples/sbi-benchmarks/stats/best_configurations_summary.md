# Best Model Configurations

Only showing parameters that vary across configuration versions for a given task/method. Rows show the best configuration version for each training method.

## Task: two_moons

### Model: Flux1

#### Budget: 10k

*No data available.*

#### Budget: 30k

*No data available.*

#### Budget: 100k

*No data available.*

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.785) | - | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.580) | - | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.573) | - | concat |

## Task: bernoulli_glm

### Model: Flux1

#### Budget: 10k

*No data available.*

#### Budget: 30k

*No data available.*

#### Budget: 100k

*No data available.*

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | training.ema_decay | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v13 (0.726) | v12 (0.862) | concat | 16 | 0.9999 | 100000 | 500 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | training.batch_size | training.val_every |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v13 (0.624) | v12 (0.893) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 256 | 500 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | training.ema_decay | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v13 (0.606) | v12 (0.618) | concat | 16 | 0.9999 | 100000 | 500 |

## Task: gaussian_linear

### Model: Flux1

#### Budget: 10k

*No data available.*

#### Budget: 30k

*No data available.*

#### Budget: 100k

*No data available.*

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps |
|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v12 (0.620) | v13 (0.723) | concat | 8 | 8 | 8 | 100000 |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v13 (0.564) | v12 (0.581) | concat | 16 | 10 | 10 | 50000 | 100 |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps |
|---|---|---|---|---|---|---|---|
| diffusion_flux1joint | v13 (0.562) | v12 (0.572) | concat | 16 | 10 | 10 | 50000 |

## Task: gaussian_mixture

### Model: Flux1

#### Budget: 10k

*No data available.*

#### Budget: 30k

*No data available.*

#### Budget: 100k

*No data available.*

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.825) | - | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.513) | - | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.516) | - | concat |

## Task: slcp

### Model: Flux1

#### Budget: 10k

*No data available.*

#### Budget: 30k

*No data available.*

#### Budget: 100k

*No data available.*

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.871) | - | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.668) | - | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| diffusion_flux1joint | v12 (0.588) | - | concat |

