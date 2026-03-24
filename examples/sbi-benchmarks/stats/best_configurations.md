# Best Configurations by C2ST Score

For each task, model type, and simulation budget, the configuration version (experiment factor) yielding the minimum C2ST score is shown, formatted as `v[exp] (C2ST score)`.

## Task: two_moons

### Model: Flux1

| Budget | flow_flux | diffusion_flux | score_matching_flux |
|---|---|---|---|
| 10k | v6 (0.5489) | v1 (0.5539) | v8 (0.5294) |
| 30k | v5 (0.5385) | v2 (0.5343) | v6 (0.5172) |
| 100k | v2 (0.5251) | v2 (0.5233) | v3 (0.5072) |

### Model: Flux1Joint

| Budget | flow_flux1joint | diffusion_flux1joint | score_matching_flux1joint |
|---|---|---|---|
| 10k | v5 (0.5462) | v8 (0.6125) | v8 (0.5222) |
| 30k | v3 (0.5286) | v8 (0.6028) | v2 (0.5129) |
| 100k | v2 (0.5242) | v8 (0.6045) | v3 (0.5023) |

## Task: bernoulli_glm

### Model: Flux1

| Budget | flow_flux | diffusion_flux | score_matching_flux |
|---|---|---|---|
| 10k | v7 (0.7218) | v7 (0.7010) | v1 (0.6923) |
| 30k | v4 (0.6959) | v1 (0.6733) | v4 (0.5790) |
| 100k | v7 (0.5504) | v7 (0.5754) | v8 (0.5456) |

### Model: Flux1Joint

| Budget | flow_flux1joint | diffusion_flux1joint | score_matching_flux1joint |
|---|---|---|---|
| 10k | v8 (0.6164) | v5 (0.8242) | v7 (0.6297) |
| 30k | v1 (0.5775) | v1 (0.7762) | v8 (0.5761) |
| 100k | v6 (0.5552) | v6 (0.7936) | v8 (0.5604) |

## Task: gaussian_linear

### Model: Flux1

| Budget | flow_flux | diffusion_flux | score_matching_flux |
|---|---|---|---|
| 10k | v6 (0.7143) | v2 (0.7184) | v6 (0.6659) |
| 30k | v4 (0.5078) | v1 (0.5264) | v5 (0.5088) |
| 100k | v1 (0.5036) | v6 (0.5228) | v2 (0.4996) |

### Model: Flux1Joint

| Budget | flow_flux1joint | diffusion_flux1joint | score_matching_flux1joint |
|---|---|---|---|
| 10k | v8 (0.5639) | v7 (0.7074) | v5 (0.5085) |
| 30k | v4 (0.5118) | v8 (0.6004) | v5 (0.5031) |
| 100k | v1 (0.4991) | v8 (0.6014) | v2 (0.5010) |

## Task: gaussian_mixture

### Model: Flux1

| Budget | flow_flux | diffusion_flux | score_matching_flux |
|---|---|---|---|
| 10k | v6 (0.5535) | v1 (0.5621) | v4 (0.5242) |
| 30k | v8 (0.5204) | v4 (0.5201) | v1 (0.5147) |
| 100k | v8 (0.5101) | v2 (0.5103) | v2 (0.5021) |

### Model: Flux1Joint

| Budget | flow_flux1joint | diffusion_flux1joint | score_matching_flux1joint |
|---|---|---|---|
| 10k | v8 (0.5221) | v1 (0.5579) | v1 (0.5105) |
| 30k | v8 (0.5155) | v4 (0.5501) | v8 (0.5037) |
| 100k | v8 (0.5127) | v3 (0.5430) | v2 (0.5008) |

## Task: slcp

### Model: Flux1

| Budget | flow_flux | diffusion_flux | score_matching_flux |
|---|---|---|---|
| 10k | v7 (0.8334) | v7 (0.8476) | v6 (0.8456) |
| 30k | v6 (0.7248) | v6 (0.7465) | v1 (0.6922) |
| 100k | v7 (0.6018) | v1 (0.6173) | v8 (0.6777) |

### Model: Flux1Joint

| Budget | flow_flux1joint | diffusion_flux1joint | score_matching_flux1joint |
|---|---|---|---|
| 10k | v7 (0.6569) | v6 (0.8262) | v1 (0.6631) |
| 30k | v8 (0.5857) | v8 (0.6803) | v8 (0.6864) |
| 100k | v8 (0.5657) | v8 (0.6550) | v2 (0.5345) |

