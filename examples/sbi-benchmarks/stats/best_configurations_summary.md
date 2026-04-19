# Best Model Configurations

Only showing parameters that vary across configuration versions for a given task/method. Rows show the best configuration version for each training method.

## Task: two_moons

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v6 (0.549) | v8 (0.552) | sum |
| diffusion_flux | v1 (0.554) | v6 (0.559) | sum |
| score_matching_flux | v8 (0.529) | v5 (0.531) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v9 (0.529) | v5 (0.538) | concat |
| diffusion_flux | v2 (0.534) | v1 (0.543) | sum |
| score_matching_flux | v9 (0.516) | v6 (0.517) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v2 (0.525) | v9 (0.526) | sum |
| diffusion_flux | v2 (0.523) | v3 (0.541) | sum |
| score_matching_flux | v3 (0.507) | v2 (0.508) | sum |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v9 (0.542) | v5 (0.546) | concat |
| diffusion_flux1joint | v12 (0.785) | - | concat |
| score_matching_flux1joint | v8 (0.522) | v4 (0.524) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v3 (0.529) | v2 (0.529) | concat |
| diffusion_flux1joint | v12 (0.580) | - | concat |
| score_matching_flux1joint | v2 (0.513) | v9 (0.518) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v2 (0.524) | v3 (0.524) | concat |
| diffusion_flux1joint | v12 (0.573) | - | concat |
| score_matching_flux1joint | v3 (0.502) | v2 (0.504) | concat |

## Task: bernoulli_glm

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v7 (0.722) | v5 (0.734) | concat |
| diffusion_flux | v7 (0.701) | v5 (0.727) | concat |
| score_matching_flux | v1 (0.692) | v4 (0.721) | sum |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v4 (0.696) | v7 (0.713) | sum |
| diffusion_flux | v1 (0.673) | v4 (0.682) | sum |
| score_matching_flux | v4 (0.579) | v7 (0.589) | sum |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v7 (0.550) | v1 (0.557) | concat |
| diffusion_flux | v7 (0.575) | v4 (0.576) | concat |
| score_matching_flux | v8 (0.546) | v6 (0.554) | concat |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | training.ema_decay | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.616) | v7 (0.645) | concat | - | - | - | - |
| diffusion_flux1joint | v13 (0.726) | v12 (0.862) | concat | 16 | 0.9999 | 100000 | 500 |
| score_matching_flux1joint | v7 (0.630) | v4 (0.662) | concat | - | - | - | - |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.params_dtype | optimizer.decay_transition | optimizer.max_lr | optimizer.min_lr | training.batch_size | training.val_every |
|---|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 (0.578) | v6 (0.583) | concat | - | - | - | - | - | - |
| diffusion_flux1joint | v13 (0.624) | v12 (0.893) | concat | bfloat16 | 0.8 | 0.0001 | 1e-06 | 256 | 500 |
| score_matching_flux1joint | v8 (0.576) | v1 (0.584) | concat | - | - | - | - | - | - |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | training.ema_decay | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|
| flow_flux1joint | v6 (0.555) | v4 (0.556) | concat | - | - | - | - |
| diffusion_flux1joint | v13 (0.606) | v12 (0.618) | concat | 16 | 0.9999 | 100000 | 500 |
| score_matching_flux1joint | v8 (0.560) | v2 (0.571) | concat | - | - | - | - |

## Task: gaussian_linear

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v6 (0.714) | v3 (0.761) | sum |
| diffusion_flux | v2 (0.718) | v3 (0.729) | concat |
| score_matching_flux | v6 (0.666) | v7 (0.820) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v4 (0.508) | v9 (0.729) | sum |
| diffusion_flux | v1 (0.526) | v9 (0.593) | sum |
| score_matching_flux | v5 (0.509) | v6 (0.512) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v9 (0.501) | v1 (0.504) | sum |
| diffusion_flux | v6 (0.523) | v1 (0.524) | concat |
| score_matching_flux | v2 (0.500) | v5 (0.500) | sum |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps |
|---|---|---|---|---|---|---|---|
| flow_flux1joint | v8 (0.564) | v7 (0.567) | concat | - | - | - | - |
| diffusion_flux1joint | v12 (0.620) | v13 (0.723) | concat | 8 | 8 | 8 | 100000 |
| score_matching_flux1joint | v5 (0.508) | v6 (0.512) | concat | - | - | - | - |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps | training.val_every |
|---|---|---|---|---|---|---|---|---|
| flow_flux1joint | v4 (0.512) | v7 (0.514) | concat | - | - | - | - | - |
| diffusion_flux1joint | v13 (0.564) | v12 (0.581) | concat | 16 | 10 | 10 | 50000 | 100 |
| score_matching_flux1joint | v9 (0.500) | v5 (0.503) | concat | - | - | - | - | - |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode | model.depth_single_blocks | model.id_emb_dim | model.val_emb_dim | training.nsteps |
|---|---|---|---|---|---|---|---|
| flow_flux1joint | v1 (0.499) | v4 (0.500) | concat | - | - | - | - |
| diffusion_flux1joint | v13 (0.562) | v12 (0.572) | concat | 16 | 10 | 10 | 50000 |
| score_matching_flux1joint | v2 (0.501) | v3 (0.503) | concat | - | - | - | - |

## Task: gaussian_mixture

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v6 (0.553) | v4 (0.554) | sum |
| diffusion_flux | v1 (0.562) | v6 (0.566) | sum |
| score_matching_flux | v4 (0.524) | v6 (0.525) | sum |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v8 (0.520) | v9 (0.523) | sum |
| diffusion_flux | v4 (0.520) | v6 (0.522) | sum |
| score_matching_flux | v9 (0.510) | v1 (0.515) | sum |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v8 (0.510) | v9 (0.516) | sum |
| diffusion_flux | v2 (0.510) | v6 (0.516) | sum |
| score_matching_flux | v2 (0.502) | v3 (0.507) | sum |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v8 (0.522) | v4 (0.527) | concat |
| diffusion_flux1joint | v12 (0.825) | - | concat |
| score_matching_flux1joint | v1 (0.510) | v4 (0.512) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v8 (0.515) | v2 (0.521) | concat |
| diffusion_flux1joint | v12 (0.513) | - | concat |
| score_matching_flux1joint | v8 (0.504) | v1 (0.508) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v8 (0.513) | v2 (0.514) | concat |
| diffusion_flux1joint | v12 (0.516) | - | concat |
| score_matching_flux1joint | v2 (0.501) | v3 (0.503) | concat |

## Task: slcp

### Model: Flux1

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v7 (0.833) | v6 (0.841) | concat |
| diffusion_flux | v7 (0.848) | v6 (0.855) | concat |
| score_matching_flux | v6 (0.846) | v7 (0.854) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v6 (0.725) | v7 (0.803) | concat |
| diffusion_flux | v9 (0.741) | v6 (0.747) | concat |
| score_matching_flux | v1 (0.692) | v6 (0.739) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux | v7 (0.602) | v4 (0.619) | concat |
| diffusion_flux | v1 (0.617) | v7 (0.676) | concat |
| score_matching_flux | v8 (0.678) | v2 (0.692) | concat |

### Model: Flux1Joint

#### Budget: 10k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v7 (0.657) | v8 (0.714) | concat |
| diffusion_flux1joint | v12 (0.871) | - | concat |
| score_matching_flux1joint | v1 (0.663) | v4 (0.677) | concat |

#### Budget: 30k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v8 (0.586) | v4 (0.596) | concat |
| diffusion_flux1joint | v12 (0.668) | - | concat |
| score_matching_flux1joint | v9 (0.568) | v8 (0.686) | concat |

#### Budget: 100k

| Method | Best | 2nd Best | model.id_merge_mode |
|---|---|---|---|
| flow_flux1joint | v8 (0.566) | v1 (0.569) | concat |
| diffusion_flux1joint | v12 (0.588) | - | concat |
| score_matching_flux1joint | v2 (0.534) | v3 (0.601) | concat |

