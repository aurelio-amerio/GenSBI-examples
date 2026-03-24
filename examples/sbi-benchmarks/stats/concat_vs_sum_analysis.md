# Concat vs Sum ID Embedding Analysis (Flux1)

Lower C2ST is better (0.5 = perfect).  
Differences ≥ 0.02 shown in **bold** (significant). Differences < 0.01 marked as ≈ equiv.

## Config Summary

| Task | #Tokens | Sum: `axes_dim` | Concat: `val_emb_dim` / `id_emb_dim` |
|---|---|---|---|
| two_moons | 2 | [10] | 10 / ? |
| bernoulli_glm | 5 | [20] | 20 / ? |
| gaussian_linear | 10 | [10] | 8 / 4 |
| gaussian_mixture | 2 | [10] | *(not tested)* |
| slcp | 5 | *(not tested)* | 20 / 10 |

Only two_moons, bernoulli_glm, and gaussian_linear have both modes tested (v1-4 = sum, v5-8 = concat).

---

## two_moons — axes_dim=10 (sum) vs val_emb=10 (concat)

Same effective value dim for both strategies.

| Method | 10k | 30k | 100k |
|---|---|---|---|
| flow_flux | ≈ equiv (0.009) | ≈ equiv (0.004) | ≈ equiv (0.010) |
| diffusion_flux | ≈ equiv (0.005) | sum (0.014) | **sum (0.021)** |
| score_matching_flux | ≈ equiv (0.007) | ≈ equiv (0.001) | ≈ equiv (0.010) |

Mostly equivalent. Only diffusion_flux at 100k shows a significant sum advantage.

---

## bernoulli_glm — axes_dim=20 (sum) vs val_emb=20 (concat)

Same effective value dim, but larger model (20 vs 10).

| Method | 10k | 30k | 100k |
|---|---|---|---|
| flow_flux | **concat (0.117)** | sum (0.017) | ≈ equiv (0.007) |
| diffusion_flux | **concat (0.128)** | **sum (0.031)** | ≈ equiv (~0) |
| score_matching_flux | **sum (0.119)** | ≈ equiv (0.010) | **concat (0.030)** |

At 10k, large differences but inconsistent across methods.

---

## gaussian_linear — axes_dim=10 (sum) vs val_emb=8 (concat)

Concat uses smaller feature dim (8) than sum (10).

| Method | 10k | 30k | 100k |
|---|---|---|---|
| flow_flux | **concat (0.047)** | **sum (0.230)** | ≈ equiv (0.011) |
| diffusion_flux | sum (0.016) | **sum (0.216)** | ≈ equiv (~0) |
| score_matching_flux | **concat (0.258)** | **concat (0.098)** | ≈ equiv (~0) |

At 30k, sum wins massively for flow/diffusion — concat's smaller val_emb_dim (8 vs 10) may be a bottleneck.

---

## Key Observations

1. **Equal feature dim + enough data (100k)**: mostly equivalent.
2. **Equal feature dim + low data (10k)**: large variance, inconsistent across methods (bernoulli_glm).
3. **Smaller concat feature dim** (gaussian_linear: val_emb=8 vs axes_dim=10): sum wins strongly at 30k for flow/diffusion.
4. **Untested**: gaussian_mixture (sum only), slcp (concat only).
5. **Method-dependent**: score_matching_flux often behaves differently from flow/diffusion.
